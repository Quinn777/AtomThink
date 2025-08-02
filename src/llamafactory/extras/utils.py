import json
import os
from tqdm import tqdm

def convert_unsupported_types_to_str(d):
    if not isinstance(d, dict):
        return d
    for key, value in d.items():
        if isinstance(value, dict):
            d[key] = convert_unsupported_types_to_str(value)
        elif not isinstance(value, (str, int, float, list, bool, type(None), tuple, set)):
            d[key] = str(value)
        elif isinstance(value, (list, tuple, set)):
            d[key] = type(value)(
                convert_unsupported_types_to_str(item) if isinstance(item, dict) else item for item in value)
    return d