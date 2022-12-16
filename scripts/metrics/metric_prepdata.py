import sys
from pathlib import Path
sys.path.append('.') # 使用 vscode 默认在工作区目录下运行 python
from vae.data.base import ImagePaths
from vae.data.utils import descale_image
from tqdm.auto import tqdm
import argparse
import numpy as np
import json
from PIL import Image
from shutil import rmtree

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--src_path",
        type=str,
        default="",
        help='txt file containing original image paths'
    )
    return parser

if __name__ == "__main__":
    parser = get_parser()
    args=parser.parse_args()
    
    src_path=args.src_path
    filenames = open(src_path, 'r').read().splitlines()
    
    size=128
    random_crop=False
    
    ds=ImagePaths(filenames, size=size, random_crop=random_crop)
    
    src_path = Path(src_path)
    image_folder = src_path.parent / 'images'
    if image_folder.exists():
        print(f'removing history...')
        rmtree(str(image_folder))
    image_folder.mkdir()
    
    print(f'save {len(ds)} images to {str(image_folder)}')
    print(f'image resolution {size}')
    
    for i, data in tqdm(enumerate(ds)):
        img=data['image']
        img=descale_image(img)
        img=(255*img).astype(np.uint8)
        Image.fromarray(img).save(str(image_folder / f'{i}.png'))
        
    args.resolution=size
    args.nums=len(ds)
    args.suffix='png'
    args.image_folder=str(image_folder)
    json.dump(vars(args), open(str(image_folder.parent / 'index.json'), 'w'))