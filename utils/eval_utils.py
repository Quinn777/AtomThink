import json
import os
from tqdm import tqdm
import time
import mmcv

def read_mmcv_config(file):
    # solve config loading conflict when multi-processes
    while True:
        config = mmcv.Config.fromfile(file)
        if len(config) == 0:
            time.sleep(0.1)
            continue
        break
    return config


def save_jsonl(path: str, data: list, ) -> None:
    with open(path, 'w', encoding='utf-8') as f:
        for line in tqdm(data, desc='save'):
            f.write(json.dumps(line, ensure_ascii=False) + '\n')


def save_json(filename, ds):
    with open(filename, 'w') as f:
        json.dump(ds, f, indent=4)


def read_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def read_jsonl(path: str, key: str = None):
    data = []
    with open(os.path.expanduser(path)) as f:
        for line in f:
            if not line:
                continue
            data.append(json.loads(line))
    if key is not None:
        data.sort(key=lambda x: x[key])
        data = {item[key]: item for item in data}
    return data