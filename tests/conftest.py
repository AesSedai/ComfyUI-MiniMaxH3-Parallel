import torch

from comfy.cli_args import args


if not torch.cuda.is_available():
    args.cpu = True
