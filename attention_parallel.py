import torch
import comfy_kitchen

from comfy.ldm.modules.attention import _comfy_kitchen_int8_inputs, attention_comfy_kitchen_int8


def balanced_head_ranges(heads, devices):
    base, extra = divmod(heads, devices)
    start = 0
    ranges = []
    for rank in range(devices):
        count = base + (rank < extra)
        ranges.append((start, start + count))
        start += count
    return ranges


def _packed_head_slice(quantized, start, end, device):
    batch, heads, _, head_dim = quantized.q.shape
    padded_length = quantized.v.shape[-1]
    v = quantized.v.view(batch, heads, head_dim, padded_length)[:, start:end]
    v_scale = quantized.v_scale.view(batch, heads, head_dim)[:, start:end]
    return comfy_kitchen.PrequantizedInt8Attention(
        q=quantized.q[:, start:end].to(device, non_blocking=True).contiguous(),
        k=quantized.k[:, start:end].to(device, non_blocking=True).contiguous(),
        v=v.reshape(-1, padded_length).to(device, non_blocking=True),
        q_scale=quantized.q_scale[:, start:end].to(device, non_blocking=True).contiguous(),
        k_scale=quantized.k_scale[:, start:end].to(device, non_blocking=True).contiguous(),
        v_scale=v_scale.reshape(-1).to(device, non_blocking=True),
        original_head_dim=quantized.original_head_dim,
        input_dtype=quantized.input_dtype,
        attention_scale=quantized.attention_scale,
        cta_k=quantized.cta_k,
        attn_mask=None,
    )


class ComfyKitchenInt8AttentionParallel:
    """Exact head-parallel CK attention using activation-only CUDA peer copies."""

    __slots__ = ("device_count", "min_sequence_length", "selected_devices", "streams")

    def __init__(self, device_count=None, min_sequence_length=32768):
        self.device_count = device_count
        self.min_sequence_length = min_sequence_length
        self.selected_devices = None
        self.streams = {}

    @staticmethod
    def _cuda_device(device):
        device = torch.device(device)
        if device.type != "cuda":
            raise RuntimeError("MiniMax H3 attention parallelism requires the model on a CUDA device")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        return device

    def devices(self, root_device):
        root_device = self._cuda_device(root_device)
        if self.selected_devices is not None and self.selected_devices[0] == root_device:
            return self.selected_devices
        if not comfy_kitchen.int8_attention_is_available(root_device):
            raise RuntimeError(f"Comfy Kitchen INT8 attention is unavailable on {root_device}")

        candidates = []
        failures = []
        for index in range(torch.cuda.device_count()):
            device = torch.device("cuda", index)
            if device == root_device:
                continue
            if not comfy_kitchen.int8_attention_is_available(device):
                failures.append(f"{device}: CK INT8 attention unavailable")
                continue
            if not torch.cuda.can_device_access_peer(root_device.index, index) or not torch.cuda.can_device_access_peer(index, root_device.index):
                failures.append(f"{device}: peer access with {root_device} unavailable")
                continue
            candidates.append(device)

        requested = min(4, torch.cuda.device_count()) if self.device_count is None else self.device_count
        devices = [root_device, *candidates[:requested - 1]]
        if self.device_count is not None and len(devices) != requested:
            details = "; ".join(failures) if failures else "not enough visible CUDA devices"
            raise RuntimeError(f"MiniMax H3 attention parallelism requested {requested} GPUs but found {len(devices)} suitable devices ({details})")
        for device in devices:
            if device not in self.streams:
                self.streams[device] = torch.cuda.Stream(device=device)
        self.selected_devices = devices
        return self.selected_devices

    def __call__(self, _, *args, **kwargs):
        kwargs["_inside_attn_wrapper"] = True
        return attention_comfy_kitchen_int8(*args, **kwargs)

    @torch.compiler.disable
    def container_function(self, q, k, v, heads, mask=None, attn_precision=None, skip_reshape=False, skip_output_reshape=False, **kwargs):
        q = q.take()
        k = k.take()
        v = v.take()
        q, k, v, mask, batch, dim_head = _comfy_kitchen_int8_inputs(
            q, k, v, heads, mask, skip_reshape, kwargs.get("enable_gqa", False)
        )
        devices = self.devices(q.device)
        if self.device_count is None and q.shape[2] < self.min_sequence_length:
            devices = devices[:1]
        if len(devices) == 1:
            quantized = comfy_kitchen.prequantize_int8_attention(
                q,
                k,
                v,
                scale=kwargs.get("scale", None),
                attn_mask=mask,
            )
            del q, k, v
            out = comfy_kitchen.int8_attention_from_prequantized(quantized)
            if not skip_output_reshape:
                out = out.transpose(1, 2).reshape(batch, -1, heads * dim_head)
            return out
        if mask is not None:
            raise RuntimeError("MiniMax H3 attention parallelism does not support attention masks")
        if q.shape[1] != heads:
            raise RuntimeError(f"attention head count is {q.shape[1]}, expected {heads}")
        if q.shape[1] != k.shape[1]:
            raise RuntimeError("MiniMax H3 attention parallelism requires equal Q, K, and V head counts")
        if len(devices) > heads:
            raise RuntimeError(f"cannot shard {heads} attention heads over {len(devices)} GPUs")

        ranges = balanced_head_ranges(heads, len(devices))
        root_device = devices[0]
        caller_stream = torch.cuda.current_stream(root_device)
        root_stream = self.streams[root_device]
        input_ready = torch.cuda.Event()
        input_ready.record(caller_stream)
        root_stream.wait_event(input_ready)
        with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
            quantized = comfy_kitchen.prequantize_int8_attention(
                q,
                k,
                v,
                scale=kwargs.get("scale", None),
                attn_mask=None,
            )
            q.record_stream(root_stream)
            k.record_stream(root_stream)
            v.record_stream(root_stream)
            output = torch.empty(
                batch,
                heads,
                quantized.q.shape[2],
                quantized.original_head_dim,
                dtype=quantized.input_dtype,
                device=root_device,
            )
            packed_ready = torch.cuda.Event()
            packed_ready.record(root_stream)
        del q, k, v

        worker_shards = {}
        worker_outputs = {}
        worker_done = {}

        try:
            for device, (start, end) in zip(devices[1:], ranges[1:]):
                with torch.cuda.device(device):
                    stream = self.streams[device]
                    stream.wait_event(packed_ready)
                    try:
                        with torch.cuda.stream(stream):
                            worker_shards[device] = _packed_head_slice(quantized, start, end, device)
                    except Exception as error:
                        raise RuntimeError(f"failed to stage CK attention heads {start}:{end} on {device}") from error

            for device, (start, end) in zip(devices[1:], ranges[1:]):
                try:
                    with torch.cuda.device(device), torch.cuda.stream(self.streams[device]):
                        worker_outputs[device] = comfy_kitchen.int8_attention_from_prequantized(worker_shards[device])
                except Exception as error:
                    raise RuntimeError(f"CK attention failed for heads {start}:{end} on {device}") from error

            start, end = ranges[0]
            with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
                local = _packed_head_slice(quantized, start, end, root_device)
                local_output = comfy_kitchen.int8_attention_from_prequantized(local)
                output[:, start:end].copy_(local_output)

            for device, (start, end) in zip(devices[1:], ranges[1:]):
                try:
                    with torch.cuda.device(device), torch.cuda.stream(self.streams[device]):
                        output[:, start:end].copy_(worker_outputs[device], non_blocking=True)
                        done = torch.cuda.Event()
                        done.record(self.streams[device])
                        worker_done[device] = done
                except Exception as error:
                    raise RuntimeError(f"failed to return CK attention heads {start}:{end} from {device}") from error

            with torch.cuda.device(root_device), torch.cuda.stream(root_stream):
                for done in worker_done.values():
                    root_stream.wait_event(done)
                if not skip_output_reshape:
                    output = output.transpose(1, 2).reshape(batch, -1, heads * dim_head)
                finished = torch.cuda.Event()
                finished.record(root_stream)
            caller_stream.wait_event(finished)
            output.record_stream(caller_stream)
            return output
        except Exception:
            for device in devices:
                try:
                    torch.cuda.synchronize(device)
                except RuntimeError:
                    pass
            raise
