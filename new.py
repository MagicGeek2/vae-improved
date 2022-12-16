import argparse
from omegaconf import OmegaConf
import json

def get_parser(**parser_kwargs):
    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "--src_path",
        type=str,
        default="",
        help='txt file containing original image paths'
    )
    return parser

parser = get_parser()
args=parser.parse_args()

# print(vars(args))
# print(OmegaConf.to_yaml(args))
json.dump(vars(args), open('tmp/index.json', 'w'))