from typing import Union
from pathlib import Path
import json


def get_dirs(home_path: Union[str, Path], file_type: str= None, is_file= True):
    if not isinstance(home_path, Path) and isinstance(home_path, str):
        home_path = Path(home_path)
    
    if file_type is not None:
        dirs_list = [sub_dir for sub_dir in home_path.iterdir() if sub_dir.suffix == file_type]
    elif is_file:
        dirs_list = [sub_dir for sub_dir in home_path.iterdir() if sub_dir.is_file()]
    else:
        dirs_list = [sub_dir for sub_dir in home_path.iterdir() if sub_dir.is_dir()]
    
    return dirs_list.sort()


def read_json(path: Union[str, Path]):
    if not isinstance(path, Path) and isinstance(path, str):
        path = Path(path)
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Union[str, Path], data: dict):
    if not isinstance(path, Path) and isinstance(path, str):
        path = Path(path)
    path.write_text(json.dumps(data, ensure_ascii=False), encoding='utf-8')