import os

import numpy as np
import ome_types
import zarr
from PIL import Image
from matplotlib import colors
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from tifffile import TiffFile

from src.app import check_ext, G, get_empty_path
from src.render_jpg import composite_channel
from src.render_png import colorize_mask


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
