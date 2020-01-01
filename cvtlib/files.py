from os import path
from glob import iglob
from typing import Tuple


def list_files(
    root_path: str,
    extensions: Tuple[str, ...],
    recursive: bool = True
):
    file_paths = []
    for ext in extensions:
        if recursive:
            root_path_ext = path.join(root_path, f'**/*{ext}')
            file_paths.extend(iglob(root_path_ext, recursive=True))
        else:
            root_path_ext = path.join(root_path, f'*{ext}')
            file_paths.extend(iglob(root_path_ext, recursive=False))

    return file_paths


def relpath(file_path, child_path):
    return path.relpath(path.join(path.dirname(file_path), child_path))

