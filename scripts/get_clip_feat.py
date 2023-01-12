import numpy as np
import torch
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from argparse import Namespace
import os
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import sys
sys.path.append('.')

import clip
from vae.util import set_seed

class SimpleDS(Dataset):
    def __init__(self, data_root, image_transform, args) -> None:
        super().__init__()
        dir_list=os.listdir(data_root)
        dir_list.sort()
        if args.end_idx > len(dir_list):
            args.end_idx = len(dir_list)
        dir_list=dir_list[args.start_idx:args.end_idx]
        print(f'handling No.{args.start_idx} to No.{args.end_idx-1} imgs')
        
        self.data_root=data_root
        self.img_names=dir_list
        
        self.image_transform=image_transform
    
    def __len__(self,):
        return len(self.img_names)
    
    def __getitem__(self, i):
        img_path = os.path.join(self.data_root, self.img_names[i])
        image = Image.open(img_path).convert("RGB")
        image=self.image_transform(image)
        return {
            'images': image, 
            'img_names': self.img_names[i],
        }


def parse_args():
    args=Namespace(
        device_num=4,
        input_dir='data/OpenImages/train',
        seed=2333,
        start_idx=0,
        end_idx=int(60_0000),
    )
    return args

@torch.no_grad()
def img2feat_and_save(images, filenames, model, output_dir, device):
    image_input = torch.tensor(np.stack(images)).to(device)
    image_features = model.encode_image(image_input).float().cpu()
    for j,fname in enumerate(filenames):
        np.save(os.path.join(output_dir, fname+'.npy'), image_features[j])

if __name__=='__main__':
    args=parse_args()
    set_seed(args.seed)

    device=f'cuda:{args.device_num}'

    # assert torch.__version__.split(".") >= ["1", "7", "1"], "PyTorch 1.7.1 or later is required"

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    # if use model name , specify download_root to your custom dir ; ex. 'ViT-B/16' , download_root='some_path'
    # or use file path directly ; ex. 'some_path/ViT-B-16.pt'
    print('Loading clip model ...')
    model, preprocess = clip.load("checkpoints/clip/ViT-B-16.pt")
    print('Done!')

    model.eval().requires_grad_(False).to(device)

    input_dir = args.input_dir
    
    output_dir = input_dir+'_clip'
    os.makedirs(output_dir, exist_ok=True)

    bs = 64
    # images = []
    # filenames = []
    
    # dir_list=os.listdir(input_dir)
    # dir_list.sort()
    # dir_list=dir_list[args.start_idx:args.end_idx]
    # print(f'handling No.{args.start_idx} to No.{args.end_idx-1} imgs')
    
    ds=SimpleDS(input_dir, preprocess, args)
    dl=DataLoader(ds, batch_size=bs, shuffle=False, num_workers=1)
    
    for batch in tqdm(dl):
        images, img_names = batch['images'], batch['img_names']
        img2feat_and_save(images, img_names, model, output_dir, device)
    
    # for i,filename in tqdm(enumerate(dir_list)):
    #     image = Image.open(os.path.join(input_dir, filename)).convert("RGB")
    #     images.append(preprocess(image))
    #     filenames.append(filename)
    #     if len(filenames)==bs:
    #         img2feat_and_save(images, filenames, model, output_dir, device)
    #         images = []
    #         filenames = []
    # if len(filenames)>0:
    #     img2feat_and_save(images, filenames, model, output_dir, device)