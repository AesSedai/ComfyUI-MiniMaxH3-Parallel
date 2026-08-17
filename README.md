# ComfyUI MiniMax H3 Parallel Attention

Exact activation-only attention-head sharding for MiniMax H3 Ref2VA. The
MiniMax model, text encoder, and VAEs stay on their normal devices; helper GPUs
receive only packed INT8 Q/K/V head slices and return BF16 attention outputs.

## Requirements

- ComfyUI 0.33.0 or newer
- Comfy Kitchen 0.2.31 or a compatible newer release with CUDA INT8 attention
- two to four visible NVIDIA CUDA GPUs with bidirectional peer access
- a MiniMax H3 model using the native ComfyUI integration

The tested system used four RTX PRO 6000 Blackwell GPUs. Other NVIDIA devices
supported by Comfy Kitchen should work, but have not been benchmarked here.

## Install

Clone the repository into `ComfyUI/custom_nodes` and restart ComfyUI:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/AesSedai/ComfyUI-MiniMaxH3-Parallel.git
```

ComfyUI Manager can install the same repository URL with **Install via Git
URL**. Public Manager/Registry listing additionally requires publishing the
repository and adding the publisher metadata assigned by the Comfy Registry.

No additional Python packages are installed; the node uses the Comfy Kitchen
version supplied by ComfyUI.

## Launch and use

Expose every participating GPU to the one ComfyUI process:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py
```

Wire the model path as:

```text
Load Diffusion Model -> MiniMax H3 Attention Parallel -> scheduler and guider
```

`devices` counts the model GPU as well as helpers. For example, `4` means one
model GPU plus three activation-only helpers. `auto` uses up to four suitable
visible GPUs and honors `min_sequence_length`; an explicit count fails early
if that many peer-accessible CK-capable GPUs are not available.

The node selects exact Comfy Kitchen INT8 attention itself, so the global
`--use-ck-attention` option is not required. Do not add `Torch Compile Model`
to this graph: measured four-step Ref2VA runs gained only about 1.5% in steady
state and diverged substantially from the eager trajectory.

## Measured reference result

At `[1, 56, 93312, 128]`, four-GPU attention measured 197.2 ms versus 534.4 ms
on one GPU (2.71x for the attention call). A full 0.7 MP, six-second Ref2VA run
with a stride-1 reference video measured 18.6-18.7 seconds per denoising step.
The four-step saved output was decoded-frame identical to the exact baseline.

Helper GPUs used about 0.78 GiB each for the 14-head split and did not load
model weights.

## Limitations

- CUDA and Comfy Kitchen INT8 attention only
- no attention masks or grouped-query head mismatch
- helpers share the process and must have bidirectional CUDA peer access
- the packed-head adapter is tested against Comfy Kitchen 0.2.31; retest exact
  output when changing Comfy Kitchen versions
