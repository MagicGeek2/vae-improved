import random

# nums=20 # just for verification
nums=30_000
src = open('data/coco2014/custom_val.txt', 'r').read().splitlines()
tgt_path = 'checkpoints/metrics/original/coco2014_val/random_30000.txt'
with open(tgt_path, 'w') as tgt_file: 
    random.shuffle(src)
    content=src[:nums]
    tgt_file.write('\n'.join(content))
print(f'write {nums} paths to {tgt_path}')