# # * preparation
# src_path=checkpoints/metrics/original/coco2014_val/random_30000.txt

# # python scripts/metrics/random_imglist.py
# # python scripts/metrics/metric_prepdata.py --src_path $src_path

# # python scripts/metrics/recon_and_save.py \
# #     --device_num 0 \
# #     --cfg_path checkpoints/vqgan/2022-12-13T09-28-55_coco2014_res128_vq_d4_n1024_noattn/config.yaml \
# #     --ckpt_path checkpoints/vqgan/2022-12-13T09-28-55_coco2014_res128_vq_d4_n1024_noattn/model.ckpt \
# #     --src_path $src_path \
# #     --image_folder checkpoints/metrics/recons/1/images

# python scripts/metrics/recon_and_save.py \
#     --device_num 1 \
#     --cfg_path checkpoints/vqgan/2022-12-15T19-22-43_coco2014_res128_vq_d4_n1024_noattn_clip/config.yaml \
#     --ckpt_path checkpoints/vqgan/2022-12-15T19-22-43_coco2014_res128_vq_d4_n1024_noattn_clip/model.ckpt \
#     --src_path $src_path \
#     --image_folder checkpoints/metrics/recons/2/images

# # * FID calculation , using two image folders containing samples and ground truth images
# gt=checkpoints/metrics/original/coco2014_val/images
# # samples=checkpoints/metrics/recons/1/images
# samples=checkpoints/metrics/recons/2/images
# device_num=1

# fidelity --gpu $device_num --fid --input1 $samples --input2 $gt

# * IS calculation
# samples=checkpoints/metrics/recons/1/images
samples=checkpoints/metrics/recons/2/images
device_num=1

fidelity --gpu $device_num --isc --input1 $samples
