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
import pathlib
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from distutils import file_util
from distutils.errors import DistutilsFileError
from enum import Enum
from functools import update_wrapper, wraps
from pathlib import Path
from threading import Timer
from typing import Callable, List, Optional

# Needed for pyinstaller
from urllib.parse import unquote

from imagecodecs import _imcd, _jpeg2k, _jpeg8, _zlib  # noqa
from numcodecs import blosc, compat_ext  # noqa

# Math tools
import numpy as np
from matplotlib import colors

# File type tools
import zarr
import ome_types
from PIL import Image
from tifffile import TiffFile
from tifffile.tifffile import TiffFileError
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator

# Web App tools
import webbrowser
import uvicorn
from fastapi import FastAPI, HTTPException, Form
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# Local sub-modules
from pyramid_assemble import main as make_ome
from render_jpg import _calculate_total_tiles, composite_channel, render_color_tiles
from render_png import colorize_integer, colorize_mask, render_tile, render_u32_tiles
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


def check_ext(path):
    base, ext1 = os.path.splitext(path)
    ext2 = os.path.splitext(base)[1]
    return ext2 + ext1


def tif_path_to_ome_path(path):
    base, ext = os.path.splitext(path)
    return f"{base}.ome{ext}"


def extract_story_json_stem(input_file):
    default_out_name = input_file.stem
    # Handle extracting the actual stem from .story.json files
    if pathlib.Path(default_out_name).suffix in [".story"]:
        default_out_name = pathlib.Path(default_out_name).stem
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
        if pathlib.Path(in_path).suffix in [".csv"]:
            try:
                file_util.copy_file(in_path, out_path)
            except DistutilsFileError as e:
                print(f"Cannot copy {in_path}")
                print(e)
        else:
            print(f"Refusing to copy non-csv infovis: {in_path}")


def get_empty_path(path):
    basename = os.path.splitext(path)[0]
    return pathlib.Path(f"{basename}_tmp.txt")


class ZarrWrapper:
    """An abstraction of Zarr that does not require indices be in a specific order."""

    def __init__(self, group, dimensions):

        self.group = group
        self.dim_list = dimensions

    def __getitem__(self, full_idx_list):
        """
        Access zarr groups as if in a standard dimension order
        Args:
            full_idx_list: level, x range, y range, z index, channel number, timestep
        """

        level = full_idx_list[0]
        key_list = ["X", "Y", "Z", "C", "T"]
        key_dict = {k: v + 1 for v, k in enumerate(key_list)}

        idx_order_list = [key_dict[key] for key in self.dim_list if key in key_dict]
        idx_value_list = tuple(full_idx_list[order] for order in idx_order_list)

        tile = self.group[level].__getitem__(idx_value_list)

        if len(tile.shape) > 2:
            # Return a 2d tile unless chosen channel has RGB
            needed_axes = 2 + int(tile.shape[2] > 1)
            tile = np.squeeze(tile, axis=tuple(range(needed_axes, len(tile.shape))))

        return tile


class Opener:
    def __init__(self, path):
        self.warning = ""
        self.path = path
        self.reader = None
        self.tilesize = 1024
        self.ext = check_ext(path)
        self.default_dtype = np.uint16

        if self.ext == ".ome.tif" or self.ext == ".ome.tiff":
            self.reader = "tifffile"
            self.io = TiffFile(self.path)
            self.ome_version = self._get_ome_version()
            if self.ome_version == 5:
                self.io = TiffFile(self.path, is_ome=False)
            self.group = zarr.open(self.io.series[0].aszarr())
            # Treat non-pyramids as groups of one array
            if isinstance(self.group, zarr.core.Array):
                root = zarr.group()
                root[0] = self.group
                self.group = root
            print("OME ", self.ome_version)
            num_channels = self.get_shape()[0]
            dimensions = self.io.series[0].get_axes()
            self.wrapper = ZarrWrapper(self.group, dimensions)

            tile_0 = self.get_tifffile_tile(num_channels, 0, 0, 0, 0, 1024)
            if tile_0 is not None:
                self.default_dtype = tile_0.dtype

            if num_channels == 3 and tile_0.dtype == "uint8":
                self.rgba = True
                self.rgba_type = "3 channel"
            elif num_channels == 1 and tile_0.dtype == "uint8":
                self.rgba = True
                self.rgba_type = "1 channel"
            else:
                self.rgba = False
                self.rgba_type = None

            print("RGB ", self.rgba)
            print("RGB type ", self.rgba_type)

        elif self.ext == ".svs":
            self.io = OpenSlide(self.path)
            self.dz = DeepZoomGenerator(
                self.io, tile_size=1024, overlap=0, limit_bounds=True
            )
            self.reader = "openslide"
            self.rgba = True
            self.rgba_type = None
            self.default_dtype = np.uint8

            print("RGB ", self.rgba)
            print("RGB type ", self.rgba_type)

        else:
            self.reader = None

    def _get_ome_version(self):
        try:
            software = self.io.pages[0].tags[305].value
            sub_ifds = self.io.pages[0].tags[330].value
            if "Faas" in software or sub_ifds is None:
                return 5
            else:
                return 6
        except Exception as e:
            print(e)
            return 5

    def read_metadata(self):
        if self.ext == ".ome.tif" or self.ext == ".ome.tiff":
            try:
                metadata = ome_types.from_tiff(self.path)
            except Exception:
                return None

            if not metadata or not metadata.images or not metadata.images[0]:
                return None

            return metadata

        return None

    def load_xml_markers(self):
        metadata = self.read_metadata()
        if not metadata:
            return []

        metadata_pixels = metadata.images[0].pixels
        if not metadata_pixels or not metadata_pixels.channels:
            return []

        return [c.name for c in metadata_pixels.channels if c.name]

    def close(self):
        self.io.close()

    def is_rgba(self, rgba_type=None):
        if rgba_type is None:
            return self.rgba
        else:
            return self.rgba and rgba_type == self.rgba_type

    def get_level_tiles(self, level, tile_size):
        if self.reader == "tifffile":

            # Negative indexing to support shape len 3 or len 2
            ny = int(np.ceil(self.group[level].shape[-2] / tile_size))
            nx = int(np.ceil(self.group[level].shape[-1] / tile_size))
            return (nx, ny)
        elif self.reader == "openslide":
            reverse_level = self.dz.level_count - 1 - level
            return self.dz.level_tiles[reverse_level]

    def get_shape(self):
        def parse_shape(shape):
            if len(shape) >= 3:
                (num_channels, shape_y, shape_x) = shape[-3:]
            else:
                (shape_y, shape_x) = shape
                num_channels = 1

            return (num_channels, shape_x, shape_y)

        if self.reader == "tifffile":

            (num_channels, shape_x, shape_y) = parse_shape(self.group[0].shape)
            all_levels = [parse_shape(v.shape) for v in self.group.values()]
            num_levels = len([shape for shape in all_levels if max(shape[1:]) > 512])
            return (num_channels, num_levels, shape_x, shape_y)

        elif self.reader == "openslide":

            (width, height) = self.io.dimensions

            def has_one_tile(counts):
                return max(counts) == 1

            small_levels = list(filter(has_one_tile, self.dz.level_tiles))
            level_count = self.dz.level_count - len(small_levels) + 1

            return (3, level_count, width, height)

    def read_tiles(self, level, channel_number, tx, ty, tilesize):
        ix = tx * tilesize
        iy = ty * tilesize

        try:
            tile = self.wrapper[
                level, ix : ix + tilesize, iy : iy + tilesize, 0, channel_number, 0
            ]
            return tile
        except Exception as e:
            G["logger"].error(e)
            return None

    def get_tifffile_tile(
        self, num_channels, level, tx, ty, channel_number, tilesize=1024
    ):

        if self.reader == "tifffile":

            tile = self.read_tiles(level, channel_number, tx, ty, tilesize)

            if tile is None:
                return np.zeros((tilesize, tilesize), dtype=self.default_dtype)

            return tile

    def get_tile(self, num_channels, level, tx, ty, channel_number, fmt=None):

        if self.reader == "tifffile":

            if self.is_rgba("3 channel"):
                tile_0 = self.get_tifffile_tile(num_channels, level, tx, ty, 0, 1024)
                tile_1 = self.get_tifffile_tile(num_channels, level, tx, ty, 1, 1024)
                tile_2 = self.get_tifffile_tile(num_channels, level, tx, ty, 2, 1024)
                tile = np.zeros((tile_0.shape[0], tile_0.shape[1], 3), dtype=np.uint8)
                tile[:, :, 0] = tile_0
                tile[:, :, 1] = tile_1
                tile[:, :, 2] = tile_2
                _format = "I;8"
            else:
                tile = self.get_tifffile_tile(
                    num_channels, level, tx, ty, channel_number, 1024
                )
                _format = fmt if fmt else "I;16"

                if _format == "RGBA" and tile.dtype != np.uint32:
                    tile = tile.astype(np.uint32)

                if _format == "I;16" and tile.dtype != np.uint16:
                    if tile.dtype == np.uint8:
                        tile = 255 * tile.astype(np.uint16)
                    else:
                        # TODO: real support for uint32, signed values, and floats
                        tile = np.clip(tile, 0, 65535).astype(np.uint16)

            return Image.fromarray(tile, _format)

        elif self.reader == "openslide":
            reverse_level = self.dz.level_count - 1 - level
            img = self.dz.get_tile(reverse_level, (tx, ty))
            return img

    def generate_mask_tiles(
        self, filename, mask_params, tile_size, level, tx, ty, should_skip_tiles={}
    ):
        num_channels = self.get_shape()[0]
        tile = self.get_tifffile_tile(num_channels, level, tx, ty, 0, tile_size)

        for image_params in mask_params["images"]:

            output_file = str(image_params["out_path"] / filename)
            if should_skip_tiles.get(output_file, False):
                continue

            target = np.zeros(tile.shape + (4,), np.uint8)
            skip_empty_tile = True

            for channel in image_params["settings"]["channels"]:
                rgba_color = [int(255 * i) for i in (colors.to_rgba(channel["color"]))]
                ids = channel["ids"]

                if len(ids) > 0:
                    bool_tile = np.isin(tile, ids)
                    # Signal that we must actually save the image
                    if not skip_empty_tile or np.any(bool_tile):
                        skip_empty_tile = False
                        target[bool_tile] = rgba_color
                else:
                    # Handle masks that color cells individually
                    target = colorize_mask(target, tile)
                    skip_empty_tile = False

            if skip_empty_tile:
                empty_file = get_empty_path(output_file)
                yield {"img": None, "empty_file": empty_file}
            else:
                img = Image.frombytes("RGBA", target.T.shape[1:], target.tobytes())
                yield {"img": img, "output_file": output_file}

    def save_mask_tiles(self, filename, mask_params, logger, tile_size, level, tx, ty):

        should_skip_tiles = {}

        for image_params in mask_params["images"]:

            output_file = str(image_params["out_path"] / filename)
            path_exists = os.path.exists(output_file) or os.path.exists(
                get_empty_path(output_file)
            )
            should_skip = path_exists and image_params.get("is_up_to_date", False)
            should_skip_tiles[output_file] = should_skip

        if all(should_skip_tiles.values()):
            logger.warning(f"Not saving tile level {level} ty {ty} tx {tx}")
            logger.warning(f"Every mask {filename} exists with same rendering settings")
            return

        if self.reader == "tifffile":
            mask_tiles = self.generate_mask_tiles(
                filename, mask_params, tile_size, level, tx, ty, should_skip_tiles
            )

            for mask_tile in mask_tiles:
                img = mask_tile.get("img", None)
                empty_file = mask_tile.get("empty_file", None)
                output_file = mask_tile.get("output_file", None)

                if all([img, output_file]):
                    img.save(output_file, compress_level=1)
                elif empty_file is not None:
                    if not os.path.exists(empty_file):
                        with open(empty_file, "w"):
                            pass

    def return_tile(self, output_file, settings, tile_size, level, tx, ty):
        if self.reader == "tifffile" and self.is_rgba("3 channel"):

            num_channels = self.get_shape()[0]
            tile_0 = self.get_tifffile_tile(num_channels, level, tx, ty, 0, tile_size)
            tile_1 = self.get_tifffile_tile(num_channels, level, tx, ty, 1, tile_size)
            tile_2 = self.get_tifffile_tile(num_channels, level, tx, ty, 2, tile_size)
            tile = np.zeros((tile_0.shape[0], tile_0.shape[1], 3), dtype=np.uint8)
            tile[:, :, 0] = tile_0
            tile[:, :, 1] = tile_1
            tile[:, :, 2] = tile_2

            return Image.fromarray(tile, "RGB")

        elif self.reader == "tifffile" and self.is_rgba("1 channel"):

            num_channels = self.get_shape()[0]
            tile = self.get_tifffile_tile(num_channels, level, tx, ty, 0, tile_size)

            return Image.fromarray(tile, "RGB")

        elif self.reader == "tifffile":
            target = None
            for i, (marker, color, start, end) in enumerate(
                zip(
                    settings["Channel Number"],
                    settings["Color"],
                    settings["Low"],
                    settings["High"],
                )
            ):
                num_channels = self.get_shape()[0]
                tile = self.get_tifffile_tile(
                    num_channels, level, tx, ty, int(marker), tile_size
                )

                if tile.dtype != np.uint16:
                    if tile.dtype == np.uint8:
                        tile = 255 * tile.astype(np.uint16)
                    else:
                        tile = tile.astype(np.uint16)

                if i == 0 or target is None:
                    target = np.zeros(tile.shape + (3,), np.float32)

                composite_channel(
                    target, tile, colors.to_rgb(color), float(start), float(end)
                )

            if target is not None:
                np.clip(target, 0, 1, out=target)
                target_u8 = (target * 255).astype(np.uint8)
                return Image.frombytes("RGB", target.T.shape[1:], target_u8.tobytes())

        elif self.reader == "openslide":
            reverse_level = self.dz.level_count - 1 - level
            return self.dz.get_tile(reverse_level, (tx, ty))

    def save_tile(self, output_file, settings, tile_size, level, tx, ty):
        img = self.return_tile(output_file, settings, tile_size, level, tx, ty)
        img.save(output_file, quality=85)


class BasicResponse(BaseModel):
    message: str


class HTTPFileNotFoundException(HTTPException):
    def __init__(self, filename):
        super(HTTPFileNotFoundException, self).__init__(
            status_code=404, detail=f"File not found: {filename}."
        )


def reset_globals():
    _g = {
        "import_pool": ThreadPoolExecutor(max_workers=1),
        "preview_cache": {},
        "image_openers": {},
        "mask_openers": {},
        "save_progress": {},
        "save_progress_max": {},
    }
    return _g


def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder at _MEIPASS
        base_path = sys._MEIPASS
    except AttributeError:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


G = reset_globals()
tiff_lock = multiprocessing.Lock()
mask_lock = multiprocessing.Lock()
app = FastAPI(title="Minerva Author")
app.mount("/static", StaticFiles(directory=resource_path("static")), name="static")

templates = Jinja2Templates(directory="templates")

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


def return_opener(path, key):
    if path not in G[key]:
        try:
            opener = Opener(path)
            return opener if opener.reader is not None else None
        except (FileNotFoundError, TiffFileError) as e:
            print(e)
            return None
    else:
        return G[key][path]


def convert_mask(path):
    ome_path = tif_path_to_ome_path(path)
    if os.path.exists(ome_path):
        return

    print(f"Converting {path}")
    tmp_dir = "minerva_author_tmp_dir"
    tmp_dir = os.path.join(os.path.dirname(path), tmp_dir)
    tmp_path = os.path.join(tmp_dir, "tmp.tif")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    make_ome([pathlib.Path(path)], pathlib.Path(tmp_path), is_mask=True, pixel_size=1)
    os.rename(tmp_path, ome_path)
    if os.path.exists(tmp_dir) and not len(os.listdir(tmp_dir)):
        os.rmdir(tmp_dir)
    print(f"Done creating {ome_path}")


def open_input_mask(path, convert=False):
    opener = None
    invalid = True
    ext = check_ext(path)
    if ext == ".ome.tif" or ext == ".ome.tiff":
        opener = return_opener(path, "mask_openers")
    elif ext == ".tif" or ext == ".tiff":
        ome_path = tif_path_to_ome_path(path)
        convertable = os.path.exists(path) and not os.path.exists(ome_path)
        if convert and convertable:
            G["import_pool"].submit(convert_mask, path)
        elif os.path.exists(ome_path):
            opener = return_opener(ome_path, "mask_openers")
            path = ome_path
        invalid = False

    success = cache_mask_opener(path, opener)
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
    if opener and not os.path.exists(opener.path):
        mask_lock.acquire()
        opener.close()
        G["mask_openers"].pop(opener.path, None)
        mask_lock.release()
        return None

    return opener


def return_mask_opener(path, convert):
    invalid = True
    if check_mask_opener(path) is None:
        invalid = open_input_mask(path, convert)
    opener = check_mask_opener(path)
    return (invalid, opener)


def return_image_opener(path):
    opener = return_opener(path, "image_openers")
    success = cache_image_opener(path, opener)
    return (not success, opener)


def nocache(view: Callable):
    @wraps(view)
    def no_cache(*args, **kwargs):
        response = view(*args, **kwargs)
        response.headers["Last-Modified"] = datetime.now()
        response.headers[
            "Cache-Control"
        ] = "no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "-1"
        return response

    return update_wrapper(no_cache, view)


def load_mask_state_subsets(filename):
    all_mask_states = {}
    path = pathlib.Path(filename)
    if not path.is_file() or path.suffix != ".csv":
        return None

    with open(path, encoding="utf-8-sig") as cf:
        state_labels = []
        for row in csv.DictReader(cf):
            if "CellID" not in row:
                print(f"No CellID found in {filename}")
                break
            try:
                cell_id = int(row.get("CellID", None))
            except TypeError:
                print(f"Cannot parse CellID in {filename}")
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
                    print(f"No State headers found in {filename}")
                    break

            # Load from each State label
            for state_i in state_labels:
                cell_state = row.get(state_i, "")
                if cell_state == "":
                    print(f'Empty {state_i} for CellID "{cell_id}" in {filename}')
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


@app.get("/", response_model=HTMLResponse)
@nocache
def root():
    """
    Serves the minerva-author web UI
    """
    return templates.TemplateResponse("index.html", {})


@app.get("/story/{session}/{file_path:path}", response_model=FileResponse)
@nocache
def out_story(session: str, file_path: str):
    """
    Serves any file path in the given story preview
    Args:
        session: unique string identifying save output
        file_path: any file path in story preview
    Returns: content of any given file
    """
    cache_dict = G["preview_cache"].get(session, {})
    path_cache = cache_dict.get(file_path, None)

    if path_cache is None:
        raise HTTPException(
            status_code=404,
            detail="Cache not found: Please restart Minerva Author to reload your save file.",
        )

    args = path_cache.get("args", [])
    kwargs = path_cache.get("kwargs", {})
    mimetype = path_cache.get("mimetype", None)
    function = path_cache.get("function", lambda: None)
    out_file = function(*args, **kwargs)

    if mimetype:
        return FileResponse(out_file, media_type=mimetype)
    else:
        return FileResponse(out_file)


class Validation(BaseModel):
    invalid: bool
    ready: bool
    path: str


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
    path = unquote(key)

    # Open the input file on the first request only
    (invalid, opener) = return_mask_opener(path, convert=True)

    return {
        "invalid": invalid,
        "ready": True if isinstance(opener, Opener) else False,
        "path": opener.path if isinstance(opener, Opener) else "",
    }


class MaskSubsets(BaseModel):
    mask_states: List[str]
    mask_subsets: List[list]
    subset_colors: List[List[int]]


@app.get("/api/mask_subsets/{key}", response_model=MaskSubsets)
@nocache
def mask_subsets(key: str):
    """
    Returns the dictionary of mask subsets
    Args:
        key: URL-escaped path to mask group csv file

    Returns: Dictionary mapping mask subsets to cell ids

    """
    path = unquote(key)

    if not os.path.exists(path):
        raise HTTPFileNotFoundException(path)

    mask_state_subsets = load_mask_state_subsets(path)
    if mask_state_subsets is None:
        raise HTTPException(status_code=404, detail=f'No mask states found at "{path}"')

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


@app.get("/api/u32/{key}/{level}_{x}_{y}.png", response_model=StreamingResponse)
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
    path = unquote(key)

    # Open the input file without allowing any conversion
    (invalid, opener) = return_mask_opener(path, convert=False)

    if isinstance(opener, Opener):
        img_io = render_tile(opener, level, x, y, 0, "RGBA")

    if img_io is None:
        raise HTTPFileNotFoundException(f"{key=}, {level=}, {x=}, {y=}.")

    return StreamingResponse(img_io, media_type="image/png")


@app.get(
    "/api/u16/{key}/{channel}/{level}_{x}_{y}.png", response_model=StreamingResponse
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
    path = unquote(key)

    # Open the input file if not already open
    (invalid, opener) = return_image_opener(path)

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

    root_dir = data["root_dir"]
    out_name = data["out_name"]

    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

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
    copy_vis_csv_files(data["waypoints"], pathlib.Path(out_dat))

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


class ProgressResponse(BaseModel):
    progress: int
    max: int


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
                    "out_path": pathlib.Path(
                        mask_path_from_index(mask_data, i, out_dir)
                    ),
                }
            )
            all_mask_params[mask_path] = mask_params
        else:
            print(f"Unable to access mask at {mask_path}")

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


def add_image_tiles_to_dict(cache_dict, config_rows, opener, out_dir_rel):
    output_path = pathlib.Path(out_dir_rel)
    ext = "jpg"

    for settings in config_rows:
        num_levels = opener.get_shape()[1]
        group_dir = settings.get("Group Path", None)
        if group_dir is None:
            print("Missing group path for image")
            continue
        # Cache tile parameters for every tile
        for level in range(num_levels):
            (nx, ny) = opener.get_level_tiles(level, 1024)
            for ty, tx in itertools.product(range(0, ny), range(0, nx)):
                filename = "{}_{}_{}.{}".format(level, tx, ty, ext)
                output_file = str(output_path / group_dir / filename)
                cache_dict[output_file] = {
                    "function": render_image_tile,
                    "mimetype": f"image/{ext}",
                    "args": [output_file, settings],
                    "kwargs": {
                        "opener": opener,
                        "tile_size": 1024,
                        "level": level,
                        "tx": tx,
                        "ty": ty,
                    },
                }

    return cache_dict


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
            print("Missing image path for mask")
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


class PreviewInput(BaseModel):
    in_file: str
    out_name: str
    groups: list
    masks: list
    waypoints: list


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
    global G

    cache_dict = {}

    path = preview_input.in_file
    out_name = preview_input.out_name
    (invalid, opener) = return_image_opener(path)
    # Ensure path is relative to output directory
    out_dir_rel = get_story_folders(out_name, "")[0]
    out_dir_rel = pathlib.Path(*pathlib.Path(out_dir_rel).parts[1:])

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
    index_filename = os.path.join(get_story_dir(), "index.html")
    cache_dict["index.html"] = {"function": lambda: index_filename}

    vis_path_dict = deduplicate_data(preview_input.waypoints, "data")
    for in_path, out_path in vis_path_dict.items():
        cache_dict[out_path] = {"function": lambda: in_path}

    cache_dict = add_mask_tiles_to_dict(cache_dict, mask_config_rows)
    cache_dict = add_image_tiles_to_dict(cache_dict, config_rows, opener, out_dir_rel)

    G["preview_cache"][session] = cache_dict
    return {"message": "OK"}


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


class GroupPath(BaseModel):
    filepath: str


class Groups(BaseModel):
    groups: dict


@app.post("/api/import/groups", response_model=Groups)
@nocache
def api_import_groups(group_path: GroupPath):
    input_file = pathlib.Path(group_path.filepath)
    if not os.path.exists(input_file):
        raise HTTPFileNotFoundException(input_file)

    saved = load_saved_file(input_file)[0]
    if not saved or "groups" not in saved:
        raise HTTPException(
            status_code=400, detail=f"File contains invalid groups: {input_file}."
        )

    return {"groups": saved["groups"]}


def load_saved_file(input_file):
    saved = None
    autosaved = None
    input_path = pathlib.Path(input_file)
    if not input_path.exists():
        return (None, None)

    if input_path.suffix == ".dat":
        saved = pickle.load(open(input_path, "rb"))
    else:
        with open(input_path) as json_file:
            saved = json.load(json_file)
            autosaved = saved.get("autosave")

    return (saved, autosaved)


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


@app.post("/api/import", response_model=ImportResponse)
@nocache
def api_import(
    filepath: str = Form(...),
    dataset: str = Form(...),
    csvpath: Optional[str] = Form(...),
    autosave_logic: Optional[str] = Form(default="skip"),
):
    response = {}
    chan_label = {}
    default_out_name = "out"
    input_file = pathlib.Path(filepath)
    input_image_file = pathlib.Path(filepath)
    loading_saved_file = input_file.suffix in [".dat", ".json"]
    root_dir = get_current_dir()

    if not os.path.exists(input_file):
        raise HTTPFileNotFoundException(input_file)

    if loading_saved_file:
        default_out_name = extract_story_json_stem(input_file)
        # autosave_logic should be "ask", "skip", or "load"
        autosave_error = autosave_logic == "ask"

        (saved, autosaved) = load_saved_file(input_file)
        root_dir = os.path.dirname(input_file)

        if is_new_autosave(saved, autosaved):
            # We need to know whether to use autosave file
            if autosave_error:
                raise HTTPException(
                    status_code=400, detail="AUTO ASK ERR: Autosave Error."
                )
            # We will load a new autosave file
            elif autosave_logic == "load":
                saved = copy_saved_states(autosaved, saved)

        input_image_file = pathlib.Path(saved["in_file"])

        if csvpath:
            csv_file = pathlib.Path(csvpath)
            if not os.path.exists(csv_file):
                raise HTTPFileNotFoundException(f'marker csv file "{csv_file}"')
        else:
            csv_file = pathlib.Path(saved["csv_file"])
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
        csv_file = pathlib.Path(csvpath)

    out_name = label_to_dir(dataset, empty=default_out_name)
    if out_name == "":
        out_name = default_out_name

    out_dir, out_yaml, out_dat, out_log = get_story_folders(out_name, root_dir)

    if not loading_saved_file and os.path.exists(out_dat):
        action = "OUT ASK ERR"
        verb = "provide an" if out_name == default_out_name else "change the"
        raise HTTPException(
            status_code=400,
            detail=f"{action}: Please {verb} output name, as {out_dat} exists.",
        )
    elif loading_saved_file and os.path.exists(out_dat):
        if not os.path.samefile(input_file, out_dat):
            action = "OUT ASK ERR"
            verb = "provide an" if out_name == default_out_name else "change the"
            command = f"Please {verb} output name or directly load {out_dat}"
            raise HTTPException(
                status_code=400,
                detail=f"{action}: {command}, as that file already exists.",
            )

    opener = None
    try:
        print("Opening file: ", str(input_image_file))

        (invalid, opener) = return_image_opener(str(input_image_file))
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

    if not os.path.exists(input_image_file):
        error_message = f"Input file {input_image_file} does not exist"
        logger.error(error_message)
        raise HTTPFileNotFoundException(input_image_file)

    return {
        "loaded": True,
        "channels": labels,
        "out_name": out_name,
        "root_dir": str(root_dir),
        "session": uuid.uuid4().hex,
        "output_save_file": str(out_dat),
        "marker_csv_file": str(csv_file),
        "input_image_file": str(input_image_file),
        "waypoints": response.get("waypoints", []),
        "sample_info": response.get(
            "sample_info", {"rotation": 0, "name": "", "text": ""}
        ),
        "masks": response.get("masks", []),
        "groups": response.get("groups", []),
        "tilesize": response.get("tilesize", 1024),
        "maxLevel": response.get("maxLevel", 1),
        "height": response.get("height", 1024),
        "width": response.get("width", 1024),
        "warning": opener.warning if opener else "",
        "rgba": opener.is_rgba() if opener else False,
    }


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


@app.get("/api/filebrowser", response_model=BrowserResponse)
@nocache
def file_browser(path: str, parent: str):
    """
    Endpoint which allows browsing the local file system

    Url parameters:
        path: path to a directory
        parent: if true, returns the contents of parent directory of given path
    Returns:
        Contents of the directory specified by path
        (or parent directory, if parent parameter is set)
    """
    folder = path
    orig_folder = folder
    if folder is None or folder == "":
        folder = Path.home()
    elif parent == "true":
        folder = Path(folder).parent

    if not os.path.exists(folder):
        raise HTTPFileNotFoundException(folder)

    response = {"entries": [], "path": str(folder)}

    # Windows: When navigating back from drive root
    # we have to show a list of available drives
    is_win_dir = os.name == "nt" and folder is not None
    if is_win_dir and str(orig_folder) == str(folder) and parent == "true":
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
    for os_stat_result in os.scandir(folder):
        try:
            is_directory = os_stat_result.is_dir()
            new_entry = {
                "name": os_stat_result.name,
                "path": os_stat_result.path,
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

    return response


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
    print("Closing tiff files")
    for opener in G["image_openers"].values():
        try:
            opener.close()
        except Exception as e:
            print(e)


def close_masks():
    print("Closing mask files")
    for opener in G["mask_openers"].values():
        try:
            opener.close()
        except Exception as e:
            print(e)


def close_import_pool():
    print("Closing import pool")
    if G["import_pool"] is not None:
        try:
            G["import_pool"].shutdown()
        except Exception as e:
            print(e)


def open_browser():
    webbrowser.open_new("http://127.0.0.1:" + str(PORT) + "/")


if __name__ == "__main__":
    Timer(1, open_browser).start()

    atexit.register(close_tiff)
    atexit.register(close_masks)
    atexit.register(close_import_pool)

    uvicorn.run("app.py:app", reload="--dev" in sys.argv, port=PORT)
