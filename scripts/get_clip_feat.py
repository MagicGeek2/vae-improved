import numpy as np
import torch
from PIL import Image
import os
from tqdm import tqdm
import sys
sys.path.append('..')

import clip


@torch.no_grad()
def img2feat_and_save(images, filenames, model, output_dir):
    image_input = torch.tensor(np.stack(images)).cuda()
    image_features = model.encode_image(image_input).float().cpu()
    for j,fname in enumerate(filenames):
        np.save(os.path.join(output_dir, fname+'.npy'), image_features[j])

if __name__=='__main__':
    device_num=0
    os.environ['CUDA_VISIBLE_DEVICES'] = f'{device_num}'

    # assert torch.__version__.split(".") >= ["1", "7", "1"], "PyTorch 1.7.1 or later is required"

    # ['RN50', 'RN101', 'RN50x4', 'RN50x16', 'ViT-B/32', 'ViT-B/16']
    # if use model name , specify download_root to your custom dir ; ex. 'ViT-B/16' , download_root='some_path'
    # or use file path directly ; ex. 'some_path/ViT-B-16.pt'
    print('Loading clip model ...')
    model, preprocess = clip.load("checkpoints/clip/ViT-B-16.pt")
    print('Done!')

    model.cuda().eval().requires_grad_(False)

    input_dir = 'data/coco2014/train2014'
    # input_dir = 'data/coco2014/val2014'
    
    output_dir = input_dir+'_clip'
    os.makedirs(output_dir, exist_ok=True)

    bs = 256
    images = []
    filenames = []
    
    dir_list=os.listdir(input_dir)
    for i,filename in tqdm(enumerate(dir_list)):
        image = Image.open(os.path.join(input_dir, filename)).convert("RGB")
        images.append(preprocess(image))
        filenames.append(filename)
        if len(filenames)==bs:
            img2feat_and_save(images, filenames, model, output_dir)
            images = []
            filenames = []
    if len(filenames)>0:
        img2feat_and_save(images, filenames, model, output_dir)