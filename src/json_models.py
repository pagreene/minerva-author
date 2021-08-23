from enum import Enum
from typing import List, Optional

from pydantic import BaseModel


class BasicResponse(BaseModel):
    message: str


class Validation(BaseModel):
    invalid: bool
    ready: bool
    path: str


class MaskSubsets(BaseModel):
    mask_states: List[str]
    mask_subsets: List[list]
    subset_colors: List[List[int]]


class SampleInfo(BaseModel):
    rotation: int
    name: str
    text: str


class SessionData(BaseModel):
    is_autosave: bool
    waypoints: list
    groups: list
    masks: list
    sample_info: SampleInfo
    csv_file: str
    in_file: str
    root_dir: str
    out_name: str


class ProgressResponse(BaseModel):
    progress: int
    max: int


class PreviewInput(BaseModel):
    in_file: str
    out_name: str
    groups: list
    masks: list
    waypoints: list


class ImageInfo(BaseModel):
    description: str


class RenderInput(BaseModel):
    in_file: str
    root_dir: str
    out_name: str
    masks: list
    groups: list
    waypoints: list
    header: str
    rotation: int
    image: ImageInfo


class GroupPath(BaseModel):
    filepath: str


class Groups(BaseModel):
    groups: dict


class ImportResponse(BaseModel):
    loaded: bool
    channels: list
    out_name: str
    root_dir: str
    session: str
    output_save_file: str
    marker_csv_file: str
    input_image_file: str
    waypoints: list
    sample_info: dict
    masks: list
    groups: list
    tilesize: int
    maxLevel: int
    height: int
    width: int
    warning: str
    rgba: bool


class AutoSaveLogic(str, Enum):
    skip = "skip"
    ask = "ask"
    load = "load"


class Entry(BaseModel):
    name: str
    path: str
    isDir: bool
    size: Optional[int]
    ctime: Optional[int]
    mtime: Optional[int]


class BrowserResponse(BaseModel):
    entries: List[Entry]
    path: str
