import os
import sys

IS_LOCAL = not os.environ.get("MINERVA_AUTHOR_NONLOCAL", False)


if not IS_LOCAL:
    from s3path import S3Path as Path
else:
    from pathlib import Path


def check_ext(path_obj: Path):
    return "".join(path_obj.suffixes)


def get_empty_path(path_obj):
    basename = os.path.splitext(path)[0]
    new_path = Path(f"{basename}_tmp.txt")
    return new_path


def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    s3_root = os.environ.get("MINERVA_AUTHOR_S3_ROOT")
    if s3_root:
        base_path = Path(f"/{s3_root}")
    else:
        try:
            # PyInstaller creates a temp folder at _MEIPASS
            base_path = sys._MEIPASS
        except AttributeError:
            base_path = Path(__file__).absolute().parent

    return base_path / relative_path
