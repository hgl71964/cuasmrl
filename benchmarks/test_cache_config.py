import os
import pickle
import argparse
from dataclasses import dataclass, field

# yapf: disable
@dataclass
class Config:
    path: str = "data"


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description="???")
    parser.add_argument('-p', "--default_out_path", type=str, dest="path", default="data")
    args = parser.parse_args()
    config = Config(**vars(args))
    return config

def main():

    config = parse_args()

    cache_path = f'{config.path}'
    with open(cache_path, 'rb') as f:
        cache_config = pickle.load(f)
    
    print(cache_config)

if __name__ == "__main__":
    main()
