import argparse
from omegaconf import OmegaConf
import json
import torch
import os
from PIL import Image
Image.MAX_IMAGE_PIXELS=None
from glob import glob
from collections import Counter
from tqdm.auto import tqdm
import imagesize

from vae.data.custom import CustomTrain, CustomTest
from vae.data.base import ImagePaths
from torch.utils.data import DataLoader

# img_dir = 'data/OpenImages/train'
# print(f'loading img paths...')
# img_paths = glob(f'{img_dir}/*.jpg')
# print(f'sorting...')
# img_paths.sort()
# print(f'done! {len(img_paths)} items in total')
# SIDE_LIMIT=1024
# start_idx, end_idx = 0, 180_0000
# if end_idx>len(img_paths):
#     end_idx=len(img_paths)
# img_paths=img_paths[start_idx:end_idx]
# print(f'handling {img_dir} No.{start_idx} to No.{end_idx-1}')

# # * 统计 open images 图片大小 (大于 1024px 的有多少)
# cnt_big=0
# for img_path in tqdm(img_paths):
#     try:
#         img_size=imagesize.get(img_path)
#     except:
#         img_size = Image.open(img_path).size
        
#     if max(img_size) > SIDE_LIMIT:
#         cnt_big+=1
# print(f'num imgs > {SIDE_LIMIT}: {cnt_big}')
# print(f'num imgs <= {SIDE_LIMIT}: {len(img_paths) - cnt_big}')

# # * 将大于 1024px 的图片 rescale 到 1024px
# def smaller_size(size, max_len):
#     ratio=max_len/max(size)
#     return int(size[0]*ratio), int(size[1]*ratio)

# cnt=0
# for img_path in tqdm(img_paths):
#     # img_size=imagesize.get(img_path)
#     try:
#         img_size=imagesize.get(img_path)
#     except:
#         img_size = Image.open(img_path).size
#     if max(img_size) > SIDE_LIMIT:
#         cnt+=1
#         img=Image.open(img_path)
#         tgt_size=smaller_size(img_size, SIDE_LIMIT)
#         # print(f'{img.one = None, with_clip_feat: bool = False) -> Nonesize} -> {size}')
#         img=img.resize(tgt_size, resample=Image.Resampling.BILINEAR)
#         img.save(img_path)
# print(f'{cnt} imgs resized')

a=(1,2)
print(a+(3,))