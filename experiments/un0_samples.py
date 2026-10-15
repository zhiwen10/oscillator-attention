"""Generate ImageNet-64 samples locally with Un-0's released 16,384-oscillator model (MIT licence).

    # download once (3.9 GB) into checkpoints/ (git-ignored):
    curl -L -o checkpoints/imagenet64_n16384.pt https://huggingface.co/un-ai/Un-0/resolve/main/imagenet64_n16384.pt
    python experiments/un0_samples.py          # -> results/un0_samples.pt, results/un0_grid.png

Samples are NOT curated: fixed seed, the first `--per-class` samples of each chosen class are kept.
torch.compile is disabled (it needs Triton, which is not available on Windows); results are unchanged.
"""
import argparse
import os

os.environ.setdefault('TORCHDYNAMO_DISABLE', '1')

from pathlib import Path

import torch
from un0.model import build_from_config, build_imagenet64_model

ROOT = Path(__file__).resolve().parents[1]
# standard ImageNet-1k class ids
CLASSES = {1: 'goldfish', 88: 'macaw', 130: 'flamingo', 207: 'golden retriever', 279: 'arctic fox',
           291: 'lion', 323: 'monarch butterfly', 388: 'giant panda', 417: 'balloon', 437: 'lighthouse',
           511: 'convertible', 948: 'Granny Smith', 970: 'alp', 973: 'coral reef', 980: 'volcano'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--checkpoint', default=str(ROOT / 'checkpoints' / 'imagenet64_n16384.pt'))
    ap.add_argument('--per-class', type=int, default=4)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model = build_from_config(build_imagenet64_model, state['config'])
    # checkpoint was saved from torch.compile'd modules; strip the wrapper prefix
    model.load_state_dict({k.replace('_orig_mod.', ''): v for k, v in state['model'].items()})
    model = model.to(dev).eval()
    n_osc = int(model.dynamics.n)
    n_dyn = sum(p.numel() for p in model.dynamics.parameters())
    n_dec = sum(p.numel() for p in model.decoder.parameters())
    print('oscillators', n_osc, '| dynamics params %.1fM | decoder params %.1fM' % (n_dyn / 1e6, n_dec / 1e6))

    ids = torch.tensor(list(CLASSES), device=dev).repeat_interleave(args.per_class)
    g = torch.Generator(device=dev).manual_seed(args.seed)
    images = []
    for chunk in ids.split(20):
        images.append(model.sample_images(chunk, generator=g).cpu())
    images = torch.cat(images)                                       # (N, 3, 64, 64) in [0, 1]
    out = ROOT / 'results' / 'un0_samples.pt'
    torch.save({'images': images, 'class_ids': ids.cpu(), 'names': CLASSES, 'per_class': args.per_class,
                'seed': args.seed, 'n_oscillators': n_osc, 'dynamics_params': n_dyn, 'decoder_params': n_dec}, out)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    k = args.per_class
    grid = images.view(len(CLASSES), k, 3, 64, 64).permute(1, 3, 0, 4, 2).reshape(k * 64, len(CLASSES) * 64, 3)
    plt.imsave(ROOT / 'results' / 'un0_grid.png', grid.numpy())
    print('saved', out, 'and results/un0_grid.png')


if __name__ == '__main__':
    main()
