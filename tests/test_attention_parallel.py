import sys
from pathlib import Path

import pytest
import torch

import comfy_kitchen  # noqa: E402
from comfy.ldm.modules.attention import AttentionTensorContainer  # noqa: E402

sys.path.insert(0, str(Path(__file__).parents[1]))
from attention_parallel import ComfyKitchenInt8AttentionParallel, balanced_head_ranges  # noqa: E402


@pytest.mark.parametrize(
    ("heads", "devices", "expected"),
    [
        (56, 2, [(0, 28), (28, 56)]),
        (56, 3, [(0, 19), (19, 38), (38, 56)]),
        (56, 4, [(0, 14), (14, 28), (28, 42), (42, 56)]),
        (7, 4, [(0, 2), (2, 4), (4, 6), (6, 7)]),
    ],
)
def test_balanced_head_ranges(heads, devices, expected):
    assert balanced_head_ranges(heads, devices) == expected


def _four_gpu_ck_available():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        return False
    for index in range(4):
        if not comfy_kitchen.int8_attention_is_available(index):
            return False
        if index and (not torch.cuda.can_device_access_peer(0, index) or not torch.cuda.can_device_access_peer(index, 0)):
            return False
    return True


@pytest.mark.skipif(not _four_gpu_ck_available(), reason="requires four peer-accessible CUDA GPUs with CK INT8 attention")
@pytest.mark.parametrize("device_count", [2, 3, 4])
def test_ck_attention_parallel_is_exact(device_count):
    generator = torch.Generator(device="cuda:0").manual_seed(1234)
    q = torch.randn((1, 7, 1024, 128), dtype=torch.bfloat16, device="cuda:0", generator=generator)
    k = torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator)
    v = torch.randn(q.shape, dtype=q.dtype, device=q.device, generator=generator)

    packed = comfy_kitchen.prequantize_int8_attention(q, k, v)
    reference = comfy_kitchen.int8_attention_from_prequantized(packed)
    parallel = ComfyKitchenInt8AttentionParallel(device_count=device_count)
    output = parallel.container_function(
        AttentionTensorContainer(q),
        AttentionTensorContainer(k),
        AttentionTensorContainer(v),
        q.shape[1],
        skip_reshape=True,
        skip_output_reshape=True,
    )
    torch.cuda.synchronize(q.device)

    assert output.shape == reference.shape
    assert output.dtype == reference.dtype
    assert output.device == reference.device
    assert torch.equal(output, reference)
