import logging

import comfy.model_base

from .attention_parallel import ComfyKitchenInt8AttentionParallel


class MiniMaxH3AttentionParallel:
    TITLE = "MiniMax H3 Attention Parallel"
    CATEGORY = "model/patch/minimax"
    FUNCTION = "patch"
    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    DESCRIPTION = "Split exact Comfy Kitchen attention heads across local GPUs without copying model weights."

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "devices": (["disabled", "auto", "2", "3", "4"], {"default": "disabled"}),
                "min_sequence_length": ("INT", {
                    "default": 32768,
                    "min": 1,
                    "max": 1000000,
                    "step": 1024,
                    "advanced": True,
                    "tooltip": "In auto mode, use one GPU below this packed token count.",
                }),
            }
        }

    def patch(self, model, devices, min_sequence_length):
        patched = model.clone()
        if devices == "disabled":
            return (patched,)
        if not isinstance(model.model, comfy.model_base.MiniMaxH3):
            raise RuntimeError("MiniMax H3 Attention Parallel requires a MiniMax H3 model")

        device_count = None if devices == "auto" else int(devices)
        parallel_attention = ComfyKitchenInt8AttentionParallel(device_count, min_sequence_length)
        selected = parallel_attention.devices(model.load_device)
        if device_count is None and len(selected) == 1:
            logging.info("MiniMax H3 attention parallelism found no suitable helper GPUs; auto will use one GPU.")
        patched.set_model_optimized_attention(parallel_attention)
        return (patched,)


NODE_CLASS_MAPPINGS = {
    "MiniMaxH3AttentionParallel": MiniMaxH3AttentionParallel,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxH3AttentionParallel": MiniMaxH3AttentionParallel.TITLE,
}
