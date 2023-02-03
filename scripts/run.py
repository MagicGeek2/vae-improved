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

#  * train
model_name = 'VQModelWithCLIP' # VQModel, VQModelWithCLIP
cfg_path = f'configs/{model_name}.yaml'
logdir=f'logs/{model_name}' 

# exp_name='OI_res256_n16384_d4' # ! change with model name and config
exp_name='OI_res256_n16384_d4_nodisc_vitB16_CLIPpreVQ' # ! change with model name and config

scale_lr='no'
debug='no'

devices = '0,1'
accelerator='gpu'

cmd=f'python main.py -t true --base {cfg_path} -n {exp_name} -l {logdir} --scale_lr {scale_lr} --debug {debug} --devices {devices} --accelerator {accelerator}'
print(cmd)
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

