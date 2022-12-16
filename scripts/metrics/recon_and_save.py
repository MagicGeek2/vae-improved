import sys
sys.path.append('.')

from vae.data.utils import descale_image
from vae.data.base import ImagePaths
from vae.models.autoencoder import VQModel

import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
import torch
from omegaconf import OmegaConf
from shutil import rmtree


def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--device_num",
        type=int,
        default=-1,
        help="-1 cpu ; >=0 specify gpu",
    )
    parser.add_argument(
        "--cfg_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--ckpt_path",
        type=str,
        default="",
    )
    parser.add_argument(
        "--src_path",
        type=str,
        default="",
        help='txt file containing original image paths'
    )
    parser.add_argument(
        "--image_folder",
        type=str,
        default="",
        help='dir to save images'
    )
    return parser


@torch.no_grad()
def reconstruct(image, model, device):
    recon = model.reconstruct(image.to(device))
    return recon


def save_recon(image, now_idx, image_folder):
    image = image.detach().cpu().numpy()
    image = descale_image(image).clip(0, 1)
    image = (255*image).astype(np.uint8).transpose(0, 2, 3, 1)
    for i, image_i in enumerate(image, start=now_idx):
        Image.fromarray(image_i).save(str(image_folder / f'{i}.png'))


if __name__ == "__main__":

    parser = get_parser()
    args=parser.parse_args()
    
    device_num = args.device_num
    device = 'cpu' if device_num == -1 else f'cuda:{device_num}'

    cfg_path = args.cfg_path
    ckpt_path = args.ckpt_path

    # load model
    params = OmegaConf.load(cfg_path)['model']['params']
    model = VQModel(params['ddconfig'], params['lossconfig'],
                    params['n_embed'], params['embed_dim']).to(device)
    model.init_from_ckpt(ckpt_path)
    model.eval().requires_grad_(False)
    print('load model done')

    # load data
    src_path = args.src_path
    filenames = open(src_path, 'r').read().splitlines()
    size = 128
    random_crop = False
    ds = ImagePaths(filenames, size=size, random_crop=random_crop)
    bs = 64
    num_workers = 0
    dl = DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers)

    # recon and save
    now_idx = 0
    image_folder = Path(args.image_folder)
    if image_folder.exists():
        print(f'removing history...')
        rmtree(str(image_folder))
    image_folder.mkdir()
    
    print(f'use reconstruction model {ckpt_path}')
    print(f'save {len(ds)} images to {str(image_folder)}')
    print(f'image resolution {size}')
    
    for batch in tqdm(dl):
        image, _ = model.get_input(batch, model.image_key)
        recon = reconstruct(image, model, device)
        save_recon(recon, now_idx, image_folder)
        now_idx += len(image)
        
    args.resolution=size
    args.nums=len(ds)
    args.suffix='png'
    json.dump(vars(args), open(str(image_folder.parent / 'index.json'), 'w'))
