import os
import json
from pathlib import Path

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def read_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def write_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)
