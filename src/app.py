import os
import re
import sys
import uuid
import time
import pickle
import string
import atexit
import csv
import io
import itertools
import json
import logging
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from distutils import file_util
from distutils.errors import DistutilsFileError
from functools import update_wrapper, wraps
from threading import Timer
from typing import Callable, Optional, List

# Needed for pyinstaller
from urllib.parse import unquote

from imagecodecs import _imcd, _jpeg2k, _jpeg8, _zlib  # noqa
from numcodecs import blosc, compat_ext  # noqa

from pydantic import BaseModel

# Web App tools
import webbrowser
import uvicorn
from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import (
    FileResponse,
    HTMLResponse,
    StreamingResponse,
    RedirectResponse,
    JSONResponse,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates

# Local sub-modules
from pyramid_assemble import main as make_ome
from render_jpg import _calculate_total_tiles, render_color_tiles
from render_png import colorize_integer, render_tile, render_u32_tiles
from opener import Opener
from exceptions import HTTPFileNotFoundException
from json_models import (
    BasicResponse,
    Validation,
    MaskSubsets,
    SessionData,
    PreviewInput,
    RenderInput,
    GroupPath,
    Groups,
    ImportResponse,
    BrowserResponse,
)
from session_manager import S3SessionManager, LocalSessionManager
from util import resource_path, check_ext, Path
from storyexport import (
    create_story_base,
    deduplicate_data,
    get_current_dir,
    get_story_dir,
    get_story_folders,
    group_path_from_label,
    label_to_dir,
    mask_label_from_index,
    mask_path_from_index,
)

if os.name == "nt":
    from ctypes import windll


PORT = 2020

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("minerva-author-app")


def tif_path_to_ome_path(path_obj: Path):
    ext = path_obj.suffix
    return path_obj.with_suffix(f".ome{ext}")


def extract_story_json_stem(input_file):
    default_out_name = input_file.stem
    # Handle extracting the actual stem from .story.json files
    if Path(default_out_name).suffix in [".story"]:
        default_out_name = Path(default_out_name).stem
    return default_out_name


def yield_labels(opener, csv_file, chan_label, num_channels):
    label_num = 0
    # First, try to load labels from CSV
    if str(csv_file) != ".":
        with open(csv_file, encoding="utf-8-sig") as cf:
            for row in csv.DictReader(cf):
                if label_num < num_channels:
                    default = row.get("marker_name", str(label_num))
                    default = row.get("Marker Name", default)
                    yield chan_label.get(str(label_num), default)
                    label_num += 1
    # Second, try to load labels from OME-XML
    else:
        for label in opener.load_xml_markers():
            yield label
            label_num += 1

    # Finally, default to numerical labels
    while label_num < num_channels:
        yield chan_label.get(str(label_num), str(label_num))
        label_num += 1


def copy_vis_csv_files(waypoint_data, json_path):
    input_dir = json_path.parent
    author_stem = extract_story_json_stem(json_path)
    vis_data_dir = f"{author_stem}-story-infovis"

    vis_path_dict_out = deduplicate_data(waypoint_data, input_dir / vis_data_dir)

    if not len(vis_path_dict_out):
        return

    if not (input_dir / vis_data_dir).exists():
        (input_dir / vis_data_dir).mkdir(parents=True)

    # Copy the visualization csv files to an infovis directory
    for in_path, out_path in vis_path_dict_out.items():
        if Path(in_path).suffix in [".csv"]:
            try:
                file_util.copy_file(in_path, out_path)
            except DistutilsFileError as e:
                logger.info(f"Cannot copy {in_path}")
                logger.exception(e)
        else:
            logger.info(f"Refusing to copy non-csv infovis: {in_path}")


if "MINERVA_AUTHOR_SESSION_STORE_ROOT" in os.environ:
    preview_cache = S3SessionManager(os.environ["MINERVA_AUTHOR_SESSION_STORE_ROOT"])
else:
    preview_cache = LocalSessionManager()


def reset_globals():
    _g = {
        "import_pool": ThreadPoolExecutor(max_workers=1),
        "image_openers": {},
        "mask_openers": {},
        "save_progress": {},
        "save_progress_max": {},
    }
    return _g


G = reset_globals()
tiff_lock = multiprocessing.Lock()
mask_lock = multiprocessing.Lock()
app = FastAPI(title="Minerva Author")
app.mount("/static", StaticFiles(directory=resource_path("static")), name="static")

templates = Jinja2Templates(directory=resource_path("templates"))

# Source: https://fastapi.tiangolo.com/tutorial/cors/
if "MINERVA_AUTHOR_CORS_ORIGINS" in os.environ:
    cors_origins = os.environ["MINERVA_AUTHOR_CORS_ORIGINS"].split(":")
else:
    cors_origins = ["http://localhost*", "https://localhost*"]

if cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


def cache_opener(path, opener, key, multi_lock):
    global G
    if isinstance(opener, Opener):
        multi_lock.acquire()
        G[key][path] = opener
        multi_lock.release()
        return True
    return False


def cache_image_opener(path, opener):
    return cache_opener(path, opener, "image_openers", tiff_lock)


def cache_mask_opener(path, opener):
    return cache_opener(path, opener, "mask_openers", mask_lock)


def convert_mask(mask_path: Path):
    # Deduce the name of the OME TIF File, return if it already exists.
    ome_path = tif_path_to_ome_path(mask_path)
    if ome_path.exists():
        return
    logger.info(f"Converting {mask_path}")

    # Define the temp directory and file path.
    tmp_path = mask_path.parent / "minerva_author_tmp_dir" / "tmp.tif"

    # Make sure the directory exists and the path does not.
    tmp_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path.unlink(missing_ok=True)

    # Make the OME TIF and move the temp to its final home.
    make_ome([mask_path], tmp_path, is_mask=True, pixel_size=1)
    tmp_path.rename(ome_path)

    # Clean up the temp directory.
    try:
        tmp_path.parent.rmdir()
    except (OSError, FileNotFoundError):
        pass
    logger.info(f"Done creating {ome_path}")


def open_input_mask(mask_path: Path, convert=False):
    opener = None
    invalid = True
    ext = check_ext(mask_path)
    if ext == ".ome.tif" or ext == ".ome.tiff":
        opener = Opener.get_opener(mask_path)
    elif ext == ".tif" or ext == ".tiff":
        ome_path = tif_path_to_ome_path(mask_path)
        convertable = mask_path.exists() and not ome_path.exists()
        if convert and convertable:
            G["import_pool"].submit(convert_mask, mask_path)
        elif ome_path.exists():
            opener = Opener.get_opener(mask_path)
            mask_path = ome_path
        invalid = False

    success = cache_mask_opener(mask_path, opener)
    return False if success else invalid


def check_mask_opener(path):
    global G
    opener = None
    ext = check_ext(path)

    if ext == ".ome.tif" or ext == ".ome.tiff":
        opener = G["mask_openers"].get(path)
    elif ext == ".tif" or ext == ".tiff":
        ome_path = tif_path_to_ome_path(path)
        opener = G["mask_openers"].get(ome_path)

    # Remove invalid openers
    if opener and not opener.path.exists():
        mask_lock.acquire()
        opener.close()
        G["mask_openers"].pop(opener.path, None)
        mask_lock.release()
        return None

    return opener


def return_mask_opener(mask_path: Path, convert: bool):
    invalid = True
    if check_mask_opener(mask_path) is None:
        invalid = open_input_mask(mask_path, convert)
    opener = check_mask_opener(mask_path)
    return invalid, opener


def return_image_opener(img_path: Path):
    opener = Opener.get_opener(img_path)
    success = cache_image_opener(img_path, opener)
    return not success, opener


def nocache(view: Callable):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = view(*args, **kwargs)
        response.headers["Last-Modified"] = str(time.time())
        response.headers[
            "Cache-Control"
        ] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "-1"
        return response

    return update_wrapper(no_cache, view)


def load_mask_state_subsets(file_path: Path):
    all_mask_states = {}
    if not file_path.is_file() or file_path.suffix != ".csv":
        return None

    with file_path.open(encoding="utf-8-sig") as cf:
        state_labels = []
        for row in csv.DictReader(cf):
            if "CellID" not in row:
                logger.info(f"No CellID found in {file_path}")
                break
            try:
                cell_id = int(row.get("CellID", None))
            except TypeError:
                logger.info(f"Cannot parse CellID in {file_path}")
                continue

            # Determine whether to use State or sequentially numbered State
            if not len(state_labels):
                state_labels = ["State"]
                if state_labels[0] not in row:
                    state_labels = []
                    for i in range(1, 10):
                        state_i = f"State{i}"
                        if state_i not in row:
                            break
                        state_labels.append(state_i)

                if not len(state_labels):
                    logger.info(f"No State headers found in {file_path}")
                    break

            # Load from each State label
            for state_i in state_labels:
                cell_state = row.get(state_i, "")
                if cell_state == "":
                    logger.info(
                        f'Empty {state_i} for CellID "{cell_id}" in {file_path}'
                    )
                    continue

                mask_subsets = all_mask_states.get(state_i, {})
                mask_group = mask_subsets.get(cell_state, set())
                mask_group.add(cell_id)

                mask_subsets[cell_state] = mask_group
                all_mask_states[state_i] = mask_subsets

    if not len(all_mask_states):
        return None

    return {
        state: {k: sorted(v) for (k, v) in mask_subsets.items()}
        for (state, mask_subsets) in all_mask_states.items()
    }


def reload_all_mask_state_subsets(masks):
    all_mask_state_subsets = {}

    def is_mask_ok(mask):
        return "map_path" in mask and "channels" in mask

    for mask in masks:
        if is_mask_ok(mask):
            all_mask_state_subsets[mask["map_path"]] = {}

    for map_path in all_mask_state_subsets:
        mask_state_subsets = load_mask_state_subsets(map_path)
        if mask_state_subsets is not None:
            all_mask_state_subsets[map_path] = mask_state_subsets

    for mask in masks:
        if not is_mask_ok(mask):
            continue

        mask_state_subsets = all_mask_state_subsets.get(mask["map_path"], {})

        # Support version 1.5.0 or lower
        mask_label = mask.get("label")
        default_label = mask.get("original_label")
        default_label = default_label if default_label else mask_label

        for chan in mask["channels"]:
            state_label = chan.get("state_label", "State")
            original_label = chan.get("original_label")
            original_label = original_label if original_label else default_label
            chan["ids"] = mask_state_subsets.get(state_label, {}).get(
                original_label, []
            )
            chan["original_label"] = original_label
            chan["state_label"] = state_label

    return masks


@app.get("/", response_class=HTMLResponse)
@nocache
def root(request: Request):
    """
    Serves the minerva-author web UI
    """
    if os.environ.get("MINERVA_AUTHOR_DEBUG", "0") == "1":
        ui_src = "http://localhost:8487/bundle.js"
    else:
        ui_src = "https://cdn.jsdelivr.net/npm/minerva-author-ui@1.0.2/build/bundle.29ba4799d2236e2efe2a.js"
    return templates.TemplateResponse(
        "index.html", {"request": request, "ui_src": ui_src}
    )


@app.get("/image/{img_path:path}")
@app.get("/images/{img_path:path}")
def get_image(img_path):
    return RedirectResponse(url=f"/static/image/{img_path}")


@app.get("/story/{session}/{file_path:path}", response_class=FileResponse)
@nocache
def out_story(session: str, file_path: str):
    """
    Serves any file path in the given story preview
    Args:
        session: unique string identifying save output
        file_path: any file path in story preview
    Returns: content of any given file
    """
    cache_dict = preview_cache.get_session(session)
    if cache_dict is None:
        raise HTTPException(
            status_code=404,
            detail="Cache not found: Please restart Minerva Author to reload your save file.",
        )

    path_cache = cache_dict.get(file_path, None)
    args = path_cache.get("args", [])
    kwargs = path_cache.get("kwargs", {})
    mimetype = path_cache.get("mimetype", None)

    # TODO: come back to this
    function = path_cache.get("function", lambda: None)
    out_file = function(*args, **kwargs)

    if mimetype:
        return FileResponse(out_file, media_type=mimetype)
    else:
        return FileResponse(out_file)


@app.get("/api/validate/u32/{key}", response_model=Validation)
@nocache
def u32_validate(key: str):
    """
    Returns status for given image mask
    Args:
        key: URL-escaped path to mask

    Returns: status dict
        invalid: whether the original path does not exist
        ready: whether the ome-tiff version of the path is ready
        path: the ome-tiff version of the path
    """
    mask_path = Path(unquote(key))

    # Open the input file on the first request only
    (invalid, opener) = return_mask_opener(mask_path, convert=True)

    return {
        "invalid": invalid,
        "ready": True if isinstance(opener, Opener) else False,
        "path": opener.path if isinstance(opener, Opener) else "",
    }


@app.get("/api/mask_subsets/{key}", response_model=MaskSubsets)
@nocache
def mask_subsets(key: str):
    """
    Returns the dictionary of mask subsets
    Args:
        key: URL-escaped path to mask group csv file

    Returns: Dictionary mapping mask subsets to cell ids

    """
    mask_path = Path(unquote(key))

    if not mask_path.exists():
        raise HTTPFileNotFoundException(mask_path)

    mask_state_subsets = load_mask_state_subsets(mask_path)
    if mask_state_subsets is None:
        raise HTTPException(
            status_code=404, detail=f'No mask states found at "{mask_path}"'
        )

    mask_states = []
    mask_subsets = []
    for (mask_state, state_subsets) in mask_state_subsets.items():
        for (k, v) in state_subsets.items():
            mask_states.append(mask_state)
            mask_subsets.append([k, v])

    return {
        "mask_states": mask_states,
        "mask_subsets": mask_subsets,
        "subset_colors": [colorize_integer(v[0]) for _, v in mask_subsets],
    }


@app.get("/api/u32/{key}/{level}_{x}_{y}.png", response_class=StreamingResponse)
@nocache
def u32_image(key: str, level: int, x: int, y: int):
    """
    Returns a 32-bit tile from given image mask
    Args:
        key: URL-escaped path to mask
        level: Pyramid level
        x: Tile coordinate x
        y: Tile coordinate y

    Returns: Tile image in png format

    """
    img_io = None
    img_path = Path(unquote(key))

    # Open the input file without allowing any conversion
    invalid, opener = return_mask_opener(img_path, convert=False)

    if isinstance(opener, Opener):
        img_io = render_tile(opener, level, x, y, 0, "RGBA")

    if img_io is None:
        raise HTTPFileNotFoundException(f"{key=}, {level=}, {x=}, {y=}.")

    return StreamingResponse(img_io, media_type="image/png")


@app.get(
    "/api/u16/{key}/{channel}/{level}_{x}_{y}.png", response_class=StreamingResponse
)
@nocache
def u16_image(key: str, channel: int, level: int, x: int, y: int):
    """
    Returns a single channel 16-bit tile from the image
    Args:
        key: URL-escaped path to image
        channel: Image channel
        level: Pyramid level
        x: Tile coordinate x
        y: Tile coordinate y

    Returns: Tile image in png format

    """
    img_io = None
    img_path = Path(unquote(key))

    # Open the input file if not already open
    invalid, opener = return_image_opener(img_path)

    if opener and not invalid:
        img_io = render_tile(opener, int(level), int(x), int(y), int(channel))

    if img_io is None:
        raise HTTPFileNotFoundException(f"{key=}, {channel=}, {level=}, {x=}, {y=}")

    return StreamingResponse(img_io, media_type="image/png")


def make_saved_chan(chan):
    # We consider ids too large to store
    return {k: v for (k, v) in chan.items() if k != "ids"}


def make_saved_mask(mask):
    new_mask = {k: v for (k, v) in mask.items() if k != "channels"}
    new_mask["channels"] = list(map(make_saved_chan, mask.get("channels", [])))
    return new_mask


def make_saved_file(data):
    new_copy = {k: v for (k, v) in data.items() if k != "masks"}
    new_copy["masks"] = list(map(make_saved_mask, data.get("masks", [])))
    return new_copy


@app.post("/api/save/{session}", response_model=BasicResponse)
@nocache
def api_save(session: str, session_data: SessionData):
    """
    Saves minerva-author project information in json file.
    Args:
        session: unique string identifying save output
        session_data: the session data given us by the server.
    Returns: OK on success

    """
    data = make_saved_file(session_data.dict())

    root_dir = Path(data["root_dir"])
    out_name = Path(data["out_name"])

    root_dir.mkdir(parents=True, exist_ok=True)

    out_dir, out_yaml, out_dat, out_log = get_story_folders(out_name, root_dir)

    saved = load_saved_file(out_dat)[0]
    # Only relegate to autosave if save file exists
    if saved and data.get("is_autosave"):
        # Copy new data to autosave and copy old saved to data
        data["autosave"] = copy_saved_states(data, {})
        data = copy_saved_states(saved, data)
        # Set the autosave timestamp
        data["autosave"]["timestamp"] = time.time()
    else:
        # Set the current timestamp
        data["timestamp"] = time.time()
        # Persist old autosaves just in case
        if saved and "autosave" in saved:
            data["autosave"] = saved["autosave"]

    with open(out_dat, "w") as out_file:
        json.dump(data, out_file)

    # Make a copy of the visualization csv files
    # for use with save_exhibit_pyramid.py
    copy_vis_csv_files(data["waypoints"], Path(out_dat))

    return {"message": "OK!"}


def render_progress_callback(current, maximum, session, key="default"):
    G["save_progress_max"][session] = G["save_progress_max"].get(session, {})
    G["save_progress"][session] = G["save_progress"].get(session, {})
    G["save_progress_max"][session][key] = maximum
    G["save_progress"][session][key] = current


def create_progress_callback(maximum, session="default", key="default"):
    def progress_callback(_current, _maximum=maximum):
        render_progress_callback(_current, _maximum, session, key)

    progress_callback(0)
    return progress_callback


@app.get("/api/render/{session}/progress")
@nocache
def get_render_progress(session: str):
    """
    Returns progress of rendering of tiles (0-100). The progress bar in minerva-author-ui uses this endpoint.
    Args:
        session: unique string identifying save output
    Returns: JSON which contains progress and max
    """
    return {
        "progress": sum(G["save_progress"].get(session, {}).values()),
        "max": sum(G["save_progress_max"].get(session, {}).values()),
    }


def format_arrow(a):
    return {
        "Text": a["text"],
        "HideArrow": a["hide"],
        "Point": a["position"],
        "Angle": 60 if a["angle"] == "" else a["angle"],
    }


def format_overlay(o):
    return {"x": o[0], "y": o[1], "width": o[2], "height": o[3]}


def make_waypoints(d, mask_data, vis_path_dict={}):

    for waypoint in d:
        mask_labels = []
        if len(mask_data) > 0:
            wp_masks = waypoint["masks"]
            mask_labels = [mask_label_from_index(mask_data, i) for i in wp_masks]
        wp = {
            "Name": waypoint["name"],
            "Description": waypoint["text"],
            "Arrows": list(map(format_arrow, waypoint["arrows"])),
            "Overlays": list(map(format_overlay, waypoint["overlays"])),
            "Group": waypoint["group"],
            "Masks": mask_labels,
            "ActiveMasks": mask_labels,
            "Zoom": waypoint["zoom"],
            "Pan": waypoint["pan"],
        }
        for vis in ["VisScatterplot", "VisCanvasScatterplot", "VisMatrix"]:
            if vis in waypoint:
                wp[vis] = waypoint[vis]
                wp[vis]["data"] = vis_path_dict[wp[vis]["data"]]

        if "VisBarChart" in waypoint:
            wp["VisBarChart"] = vis_path_dict[waypoint["VisBarChart"]]

        yield wp


def make_stories(d, mask_data=[], vis_path_dict={}):
    return [
        {
            "Name": "",
            "Description": "",
            "Waypoints": list(make_waypoints(d, mask_data, vis_path_dict)),
        }
    ]


def make_mask_yaml(mask_data):
    for (i, mask) in enumerate(mask_data):
        yield {
            "Path": mask_path_from_index(mask_data, i),
            "Name": mask_label_from_index(mask_data, i),
            "Colors": [c["color"] for c in mask["channels"]],
            "Channels": [c["label"] for c in mask["channels"]],
        }


def make_group_path(groups, group):
    c_path = "--".join(
        str(c["id"]) + "__" + label_to_dir(c["label"]) for c in group["channels"]
    )
    g_path = group_path_from_label(groups, group["label"])
    return g_path + "_" + c_path


def make_groups(d):
    for group in d:
        yield {
            "Name": group["label"],
            "Path": make_group_path(d, group),
            "Colors": [c["color"] for c in group["channels"]],
            "Channels": [c["label"] for c in group["channels"]],
        }


def make_rows(d):
    for group in d:
        channels = group["channels"]
        yield {
            "Group Path": make_group_path(d, group),
            "Channel Number": [str(c["id"]) for c in channels],
            "Low": [int(65535 * c["min"]) for c in channels],
            "High": [int(65535 * c["max"]) for c in channels],
            "Color": ["#" + c["color"] for c in channels],
        }


def make_mask_rows(out_dir, mask_data, session):
    all_mask_params = {}

    for (i, mask) in enumerate(mask_data):

        mask_params = {"opener": None, "images": []}
        mask_path = mask["path"]

        if mask_path in all_mask_params:
            mask_params = all_mask_params[mask_path]
        else:
            # Open the input file without allowing any conversion
            (invalid, mask_opener) = return_mask_opener(mask_path, convert=False)
            mask_params["opener"] = mask_opener

        if isinstance(mask_params["opener"], Opener):
            mask_opener = mask_params["opener"]
            num_levels = mask_opener.get_shape()[1]
            mask_total = _calculate_total_tiles(mask_opener, 1024, num_levels)
            mask_params["images"].append(
                {
                    "settings": {
                        "channels": [
                            {"ids": c["ids"], "color": "#" + c["color"]}
                            for c in mask["channels"]
                        ],
                        "source": str(mask_path),
                    },
                    "progress": create_progress_callback(mask_total, session, str(i)),
                    "out_path": Path(mask_path_from_index(mask_data, i, out_dir)),
                }
            )
            all_mask_params[mask_path] = mask_params
        else:
            logger.info(f"Unable to access mask at {mask_path}")

    return all_mask_params.values()


def write_json_file(data):
    bytes_io = io.BytesIO()
    data_bytes = str.encode(json.dumps(data))
    bytes_io.write(data_bytes)
    bytes_io.seek(0)
    return bytes_io


def make_exhibit_config(opener, out_name, data):

    mask_data = data["masks"]
    group_data = data["groups"]
    waypoint_data = data["waypoints"]
    vis_path_dict = deduplicate_data(waypoint_data, "data")

    (num_channels, num_levels, width, height) = opener.get_shape()

    _config = {
        "Images": [
            {
                "Name": "i0",
                "Description": data["image"]["description"],
                "Path": "images/" + out_name,
                "Width": width,
                "Height": height,
                "MaxLevel": num_levels - 1,
            }
        ],
        "Header": data["header"],
        "Rotation": data["rotation"],
        "Layout": {"Grid": [["i0"]]},
        "Stories": make_stories(waypoint_data, mask_data, vis_path_dict),
        "Masks": list(make_mask_yaml(mask_data)),
        "Groups": list(make_groups(group_data)),
    }
    return _config


def render_image_tile(output_file, settings, **kwargs):
    tile_size = kwargs.get("tile_size", 1024)
    level = kwargs.get("level", 0)
    tx = kwargs.get("tx", 0)
    ty = kwargs.get("ty", 0)
    opener = kwargs["opener"]
    img = opener.return_tile(output_file, settings, tile_size, level, tx, ty)
    img_io = io.BytesIO()
    img.save(img_io, "JPEG", quality=85)
    img_io.seek(0)
    return img_io


class CacheEntry(BaseModel):
    data: str
    mimetype: str
    args: List[str]
    kwargs: dict


def add_image_tiles_to_dict(config_rows, opener, output_path: Path):
    ext = "jpg"

    new_entries = {}

    for settings in config_rows:
        num_levels = opener.get_shape()[1]
        group_dir = settings.get("Group Path", None)
        if group_dir is None:
            logger.info("Missing group path for image")
            continue
        # Cache tile parameters for every tile
        for level in range(num_levels):
            (nx, ny) = opener.get_level_tiles(level, 1024)
            for ty, tx in itertools.product(range(0, ny), range(0, nx)):
                filename = f"{level}_{tx}_{ty}.{ext}"
                output_file = str(output_path / group_dir / filename)
                new_entries[output_file] = CacheEntry(
                    data=None,
                    mimetype=f"image/{ext}",
                    args=[output_file, settings],
                    kwargs={
                        "opener": opener,
                        "tile_size": 1024,
                        "level": level,
                        "tx": tx,
                        "ty": ty,
                    },
                )
    return new_entries


def render_mask_tile(filename, mask_params, **kwargs):
    tile_size = kwargs.get("tile_size", 1024)
    level = kwargs.get("level", 0)
    tx = kwargs.get("tx", 0)
    ty = kwargs.get("ty", 0)
    opener = mask_params["opener"]
    # We except the mask params to only contain one image
    mask_tiles = opener.generate_mask_tiles(
        filename, mask_params, tile_size, level, tx, ty
    )
    img = next(mask_tiles, {}).get("img", None)
    img_io = io.BytesIO()
    if img is not None:
        img.save(img_io, "PNG", compress_level=1)
    img_io.seek(0)
    return img_io


def add_mask_tiles_to_dict(cache_dict, mask_config_rows):
    all_mask_params = []
    ext = "png"
    # Mask params must by no longer optimized for saving
    for mask_params in mask_config_rows:
        # Unpack all images from all mask params
        for image_params in mask_params.get("images", []):
            mask_params_copy = {
                "opener": mask_params["opener"],
                "images": [image_params],
            }
            all_mask_params.append(mask_params_copy)

    for mask_params in all_mask_params:
        opener = mask_params["opener"]
        num_levels = opener.get_shape()[1]
        image_params = mask_params.get("images", [None])[0]
        output_path = image_params.get("out_path", None)
        if not all([image_params, output_path]):
            logger.info("Missing image path for mask")
            continue
        # Cache tile parameters for every tile
        for level in range(num_levels):
            (nx, ny) = opener.get_level_tiles(level, 1024)
            for ty, tx in itertools.product(range(0, ny), range(0, nx)):
                filename = "{}_{}_{}.{}".format(level, tx, ty, ext)
                output_file = str(output_path / filename)
                cache_dict[output_file] = {
                    "function": render_mask_tile,
                    "mimetype": f"image/{ext}",
                    "args": [filename, mask_params],
                    "kwargs": {"tile_size": 1024, "level": level, "tx": tx, "ty": ty},
                }

    return cache_dict


@app.post("/api/preview/{session}", response_model=BasicResponse)
@nocache
def api_preview(session: str, preview_input: PreviewInput):
    """
    Caches all preview parameters for given session
    Args:
        session: unique string identifying save output
        preview_input: the data to be previewed.
    Returns: OK on success

    """
    cache_dict = {}

    path = preview_input.in_file
    out_name = preview_input.out_name
    (invalid, opener) = return_image_opener(path)
    # Ensure path is relative to output directory
    out_dir_rel = get_story_folders(out_name, "")[0]
    out_dir_rel = Path(*Path(out_dir_rel).parts[1:])

    if invalid or not opener:
        raise HTTPFileNotFoundException(path)

    config_rows = list(make_rows(preview_input.groups))
    mask_config_rows = list(make_mask_rows(out_dir_rel, preview_input.masks, session))
    exhibit_config = make_exhibit_config(opener, out_name, preview_input.dict())
    cache_dict["exhibit.json"] = {
        "function": write_json_file,
        "args": [exhibit_config],
        "mimetype": "text/json",
    }
    index_filename = get_story_dir() / "index.html"
    cache_dict["index.html"] = {"function": lambda: index_filename}

    vis_path_dict = deduplicate_data(preview_input.waypoints, "data")
    for in_path, out_path in vis_path_dict.items():
        cache_dict[out_path] = {"function": lambda: in_path}

    cache_dict = add_mask_tiles_to_dict(cache_dict, mask_config_rows)
    cache_dict = add_image_tiles_to_dict(cache_dict, config_rows, opener, out_dir_rel)

    preview_cache.update_session(session, cache_dict)
    return {"message": "OK"}


@app.post("/api/render/{session}", response_model=BasicResponse)
@nocache
def api_render(session: str, render_input: RenderInput):
    """
    Renders all image tiles and saves them under new minerva-story instance.
    Args:
        session: unique string identifying save output
        render_input: the data given to render the image.
    Returns: OK on success

    """
    G["save_progress"] = {}
    G["save_progress_max"] = {}

    path = render_input.in_file
    root_dir = render_input.root_dir
    out_name = render_input.out_name

    (invalid, opener) = return_image_opener(path)
    out_dir, out_yaml, out_dat, out_log = get_story_folders(out_name, root_dir)

    if invalid or not opener:
        raise HTTPFileNotFoundException(path)

    data = render_input.groups
    mask_data = render_input.masks
    waypoint_data = render_input.waypoints
    config_rows = list(make_rows(data))
    create_story_base(out_name, waypoint_data, mask_data, folder=root_dir)
    exhibit_config = make_exhibit_config(opener, out_name, render_input.dict())

    with open(out_yaml, "w") as wf:
        json.dump(exhibit_config, wf)

    mask_config_rows = make_mask_rows(out_dir, mask_data, session)

    # Render all uint16 image channels
    render_color_tiles(
        opener,
        out_dir,
        1024,
        config_rows,
        logger,
        progress_callback=create_progress_callback(0, session),
    )

    # Render all uint32 segmentation masks
    for mask_params in mask_config_rows:
        render_u32_tiles(mask_params, 1024, logger)

    return {"message": "OK"}


@app.post("/api/import/groups", response_model=Groups)
@nocache
def api_import_groups(group_path: GroupPath):
    input_file = Path(group_path.filepath)
    if not input_file.exists():
        raise HTTPFileNotFoundException(input_file)

    saved = load_saved_file(input_file)[0]
    if not saved or "groups" not in saved:
        raise HTTPException(
            status_code=400, detail=f"File contains invalid groups: {input_file}."
        )

    return {"groups": saved["groups"]}


def load_saved_file(input_file):
    autosaved = None
    input_path = Path(input_file)
    if not input_path.exists():
        return None, None

    if input_path.suffix == ".dat":
        saved = pickle.load(open(input_path, "rb"))
    else:
        with open(input_path) as json_file:
            saved = json.load(json_file)
            autosaved = saved.get("autosave")

    return saved, autosaved


def copy_saved_states(from_save, to_save):
    saved_keys = [
        "sample_info",
        "waypoints",
        "groups",
        "masks",
        "in_file",
        "csv_file",
        "root_dir",
    ]
    for saved_key in saved_keys:
        if saved_key in from_save:
            to_save[saved_key] = from_save[saved_key]

    return to_save


def is_new_autosave(saved, autosaved):
    if saved is None or autosaved is None:
        return False

    autosaved_time = autosaved.get("timestamp")
    saved_time = saved.get("timestamp")
    if autosaved_time:
        if saved_time:
            # Decide if new autosave
            return autosaved_time > saved_time
        else:
            # Save file from before v1.6.0
            return True
    else:
        # Malformed autosave
        return False


class ImportData(BaseModel):
    filepath: str
    csvpath: str
    dataset: str
    autosave_logic: str


@app.post("/api/import", response_model=ImportResponse)
def api_import(import_data: ImportData):
    response = {}
    chan_label = {}
    default_out_name = "out"
    input_file = Path(import_data.filepath)
    input_image_file = Path(import_data.filepath)
    loading_saved_file = input_file.suffix in [".dat", ".json"]
    root_dir = get_current_dir()

    if not input_file.exists():
        raise HTTPFileNotFoundException(input_file)

    if loading_saved_file:
        default_out_name = extract_story_json_stem(input_file)
        # autosave_logic should be "ask", "skip", or "load"
        autosave_error = import_data.autosave_logic == "ask"

        (saved, autosaved) = load_saved_file(input_file)
        root_dir = input_file.parent

        if is_new_autosave(saved, autosaved):
            # We need to know whether to use autosave file
            if autosave_error:
                raise HTTPException(
                    status_code=400, detail="AUTO ASK ERR: Autosave Error."
                )
            # We will load a new autosave file
            elif import_data.autosave_logic == "load":
                saved = copy_saved_states(autosaved, saved)

        input_image_file = Path(saved["in_file"])

        if import_data.csvpath:
            csv_file = Path(import_data.csvpath)
            if not csv_file.exists():
                raise HTTPFileNotFoundException(f'marker csv file "{csv_file}"')
        else:
            csv_file = Path(saved["csv_file"])
        if "sample_info" in saved:
            response["sample_info"] = saved["sample_info"]
            if "rotation" not in response["sample_info"]:
                response["sample_info"]["rotation"] = 0

        if "masks" in saved:
            # This step could take up to a minute
            response["masks"] = reload_all_mask_state_subsets(saved["masks"])

        response["waypoints"] = saved["waypoints"]
        response["groups"] = saved["groups"]
        for group in saved["groups"]:
            for chan in group["channels"]:
                chan_label[str(chan["id"])] = chan["label"]
    else:
        csv_file = Path(import_data.csvpath)

    out_name = label_to_dir(import_data.dataset, empty=default_out_name)
    if out_name == "":
        out_name = default_out_name

    out_dir, out_yaml, out_dat, out_log = get_story_folders(out_name, root_dir)

    if not loading_saved_file and out_dat.exists():
        action = "OUT ASK ERR"
        verb = "provide an" if out_name == default_out_name else "change the"
        raise HTTPException(
            status_code=400,
            detail=f"{action}: Please {verb} output name, as {out_dat} exists.",
        )
    elif loading_saved_file and out_dat.exists():
        if not input_file.samefile(out_dat):
            action = "OUT ASK ERR"
            verb = "provide an" if out_name == default_out_name else "change the"
            command = f"Please {verb} output name or directly load {out_dat}"
            raise HTTPException(
                status_code=400,
                detail=f"{action}: {command}, as that file already exists.",
            )

    opener = None
    try:
        logger.info("Opening file: ", str(input_image_file))

        (invalid, opener) = return_image_opener(input_image_file)
        if invalid or not opener:
            raise HTTPFileNotFoundException(input_image_file)

        (num_channels, num_levels, width, height) = opener.get_shape()

        response["maxLevel"] = num_levels - 1
        response["tilesize"] = opener.tilesize
        response["height"] = height
        response["width"] = width

    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Invalid tiff file.")

    try:
        labels = list(yield_labels(opener, csv_file, chan_label, num_channels))
    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500, detail="Error in loading channel marker names."
        )

    fh = logging.FileHandler(str(out_log))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    if not input_image_file.exists():
        error_message = f"Input file {input_image_file} does not exist"
        logger.error(error_message)
        raise HTTPFileNotFoundException(input_image_file)

    return ImportResponse(
        loaded=True,
        channels=labels,
        out_name=out_name,
        root_dir=str(root_dir),
        session=uuid.uuid4().hex,
        output_save_file=str(out_dat),
        marker_csv_file=str(csv_file),
        input_image_file=str(input_image_file),
        waypoints=response.get("waypoints", []),
        sample_info=response.get(
            "sample_info", {"rotation": 0, "name": "", "text": ""}
        ),
        masks=response.get("masks", []),
        groups=response.get("groups", []),
        tilesize=response.get("tilesize", 1024),
        maxLevel=response.get("maxLevel", 1),
        height=response.get("height", 1024),
        width=response.get("width", 1024),
        warning=opener.warning if opener else "",
        rgba=opener.is_rgba() if opener else False,
    )


@app.get("/api/filebrowser", response_model=BrowserResponse)
@nocache
def file_browser(path: Optional[Path] = None, parent: Optional[bool] = False):
    """
    Endpoint which allows browsing the local file system

    Url parameters:
        path: path to a directory
        parent: if true, returns the contents of parent directory of given path
    Returns:
        Contents of the directory specified by path
        (or parent directory, if parent parameter is set)
    """
    print(path, parent)
    folder = path.absolute()
    orig_folder = folder
    if folder is None:
        folder = Path.home()
    elif parent:
        print("hi")
        folder = folder.parent

    print(folder)

    if not folder.exists():
        raise HTTPFileNotFoundException(folder)

    response = {"entries": [], "path": str(folder)}

    # Windows: When navigating back from drive root
    # we have to show a list of available drives
    is_win_dir = os.name == "nt" and folder is not None
    if is_win_dir and str(orig_folder) == str(folder) and parent:
        match = re.search("[A-Za-z]:\\\\$", str(folder))  # C:\ or D:\ etc.
        if match:
            drives = _get_drives_win()
            for drive in drives:
                new_entry = {
                    "name": drive + ":\\",
                    "path": drive + ":\\",
                    "isDir": True,
                }
                response["entries"].append(new_entry)
            return response

    # Return a list of folders and files within the requested folder
    for os_stat_result in folder.iterdir():
        try:
            is_directory = os_stat_result.is_dir()
            new_entry = {
                "name": os_stat_result.name,
                "path": str(os_stat_result),
                "isDir": is_directory,
            }

            is_broken = False
            is_hidden = os_stat_result.name[0] == "."

            if not is_directory:
                try:
                    stat_result = os_stat_result.stat()
                    new_entry["size"] = stat_result.st_size
                    new_entry["ctime"] = stat_result.st_ctime
                    new_entry["mtime"] = stat_result.st_mtime
                except FileNotFoundError:
                    is_broken = True

            if not is_hidden and not is_broken:
                response["entries"].append(new_entry)
        except PermissionError:
            pass

    return JSONResponse(response)


def _get_drives_win():
    """
    Returns a list of drive letters in Windows
    https://stackoverflow.com/a/827398
    """
    drives = []
    bitmask = windll.kernel32.GetLogicalDrives()
    for letter in string.ascii_uppercase:
        if bitmask & 1:
            drives.append(letter)
        bitmask >>= 1

    return drives


def close_tiff():
    logger.info("Closing tiff files")
    for opener in G["image_openers"].values():
        try:
            opener.close()
        except Exception as e:
            logger.exception(e)


def close_masks():
    logger.info("Closing mask files")
    for opener in G["mask_openers"].values():
        try:
            opener.close()
        except Exception as e:
            logger.exception(e)


def close_import_pool():
    logger.info("Closing import pool")
    if G["import_pool"] is not None:
        try:
            G["import_pool"].shutdown()
        except Exception as e:
            logger.exception(e)


def open_browser():
    webbrowser.open_new("http://127.0.0.1:" + str(PORT) + "/")


if __name__ == "__main__":
    Timer(1, open_browser).start()

    atexit.register(close_tiff)
    atexit.register(close_masks)
    atexit.register(close_import_pool)

    uvicorn.run("app:app", reload="--dev" in sys.argv, port=PORT)
