"""公开副本的数据路径接口：显式配置、固定资产键、禁止私有位置后备。"""
from pathlib import Path
import csv
import hashlib
import json
import os

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / '00_admin/shared_source_registry.csv'


def load_configuration():
    """配置是JSON；相对路径相对于配置文件，而不是当前目录。"""
    name = os.environ.get('WF_PATHS_FILE')
    if not name:
        raise FileNotFoundError('Mode B requires WF_PATHS_FILE pointing to your local JSON configuration; see config/paths.example.json. Mode A needs no data configuration.')
    path = Path(name).expanduser().resolve(strict=True)
    data = json.loads(path.read_text(encoding='utf-8'))
    if not isinstance(data, dict) or not isinstance(data.get('assets', {}), dict):
        raise ValueError('Configuration must be an object with an assets mapping.')
    return path, data


def configured_root():
    """只使用显式配置的数据根。"""
    path, data = load_configuration()
    value = data.get('data_root')
    if not value:
        raise ValueError('Set data_root in the local configuration.')
    root = Path(value).expanduser()
    return (path.parent / root).resolve() if not root.is_absolute() else root.resolve()


def asset_record(key):
    with REGISTRY.open(encoding='utf-8-sig', newline='') as f:
        records = [r for r in csv.DictReader(f) if r['asset_key'] == key]
    if len(records) != 1:
        raise KeyError(f'Unknown or non-unique registered source key: {key}')
    return records[0]


def resolve_asset(key):
    """不替换版本；数据和codebook必须位于Git工作区之外。"""
    record = asset_record(key)
    cfg_path, cfg = load_configuration()
    explicit = cfg.get('assets', {}).get(key)
    if explicit:
        target = Path(explicit).expanduser()
        if not target.is_absolute():
            target = cfg_path.parent / target
    else:
        target = configured_root() / record['relative_path']
    target = target.resolve()
    if target.is_relative_to(ROOT.resolve()):
        raise ValueError('Restricted inputs must be outside the publication repository.')
    if not target.is_file():
        raise FileNotFoundError(f'Required registered input {key} is unavailable. Configure that exact version in WF_PATHS_FILE; no fallback search is performed.')
    return target


def verify_asset(key):
    """与原来源登记的真实SHA核对；不生成或伪造替代freeze。"""
    record = asset_record(key)
    path = resolve_asset(key)
    digest = hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()
    if digest != record['sha256']:
        raise ValueError(f'{key}: source SHA-256 differs from the registered study version; review version identity before analysis.')
    return {'asset_key': key, 'sha256': digest, 'status': 'PASS'}


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Check authorized source identities without fitting models.')
    parser.add_argument('--verify-hashes', action='store_true')
    args = parser.parse_args()
    keys = ['hrs_rand_longitudinal', 'elsa_gateway_h', 'klosa_gateway_f']
    result = [verify_asset(k) if args.verify_hashes else {'asset_key': k, 'available': bool(resolve_asset(k))} for k in keys]
    print(json.dumps(result, indent=2))
