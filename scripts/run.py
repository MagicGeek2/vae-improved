import os
from time import sleep
from tqdm import tqdm


def sleep_and_tell(seconds):
    print(f'start sleeping')
    for i in tqdm(range(1, seconds+1), desc='sleeping', unit='second'):
        sleep(1)
        # print(f'sleeping now. seconds [{i}/{seconds}]')
    print(f'now I wake up')

delay_hours = 0  # 0 by default
delay_seconds = int(delay_hours * 3600)
sleep_and_tell(delay_seconds)

# * train
cfg_path = 'configs/VQModelWithCLIP.yaml'
name='OI_res256_vq_n16384_d4_nodisc_vitB16_CLIPpreVQ'
# name='test_exp'
logdir='logs/VQModelWithCLIP' # VQModel, VQModelWithCLIP
scale_lr='no'
debug='no'

devices = '0'
accelerator='gpu'
strategy=None # None, 'ddp' , 'ddp_spawn' , 
profiler=None # None, 'simple', 'advanced'

cmd=f'python main.py -t true --base {cfg_path} -n {name} -l {logdir} --scale_lr {scale_lr} --debug {debug} --devices {devices} --accelerator {accelerator}'
if strategy:
    cmd = cmd + f' --strategy {strategy}'
if profiler:
    cmd = cmd + f' --profiler {profiler}'
os.system(cmd)




# # * resume
# # # * set resume configs
# resume_path = 'logs/2022-10-17T02-26-48_res64_f4_n256_d3_noclip/checkpoints/epoch=000099.ckpt'
# config_path = 'configs/brid_clippre/res64_f4_n256_d3_noclip.yaml'
# max_epochs = None

# # # * start running command
# command = f'python main.py --resume {resume_path} --gpus {gpus} -t true'
# if config_path is not None:
#     command = command+f' --base {config_path}'
# if max_epochs is not None:
#     command = command+f' --max_epochs {max_epochs}'
# os.system(command)

