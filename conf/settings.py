# settings.py

import json
import sys
sys.path.insert(0,'../')


def build_config(config_file):
    config = {
        "sample_data": "db/sample.csv",
        "data_path": "db/2021-2024.3",
        "datasets": "db/datasets",
        "underlying": "db/akdata",
        "underlying_index": {
            "IO": "沪深300",
            "MO": "中证1000",
            "HO": "上证50"
        }
    }
    with open(config_file, 'w') as config_file:
        json.dump(config, config_file, indent=4)


def load_config(file='conf/config.json'):
    with open(file, 'r') as config_file:
        config = json.load(config_file)
    return config
