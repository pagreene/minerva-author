import logging
from io import BytesIO
from typing import Optional, Iterator

import numpy as np
import ome_types
import zarr
from PIL import Image
from matplotlib import colors
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from tifffile import TiffFile

from render_jpg import composite_channel
from render_png import colorize_mask
from util import check_ext, get_empty_path, Path

logger = logging.getLogger("minerva-author-opener")


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
    def __init__(self, path: Path):
        self.warning = ""
        self.path = path
        self.tilesize = 1024
        self.ext = check_ext(path)
        (
            self.io,
            self.reader,
            self.rgba,
            self.rgba_type,
            self.default_dtype,
        ) = self._setup()
        logger.info("RGB ", self.rgba)
        logger.info("RGB type ", self.rgba_type)

    @staticmethod
    def get_opener(path):
        ext = check_ext(path)
        if ext == ".ome.tif" or ext == ".ome.tiff":
            return TiffOpener(path)
        elif ext == ".svs":
            return SvsOpener(path)
        else:
            return

    def _setup(self):
        raise NotImplementedError

    def read_metadata(self):
        return None

    def is_rgba(self, rgba_type=None):
        if rgba_type is None:
            return self.rgba
        else:
            return self.rgba and rgba_type == self.rgba_type

    def get_level_tiles(self, level, tile_size):
        raise NotImplementedError

    def get_shape(self):
        raise NotImplementedError

    def get_tile(self, num_channels, level, tx, ty, channel_number, fmt=None):
        raise NotImplementedError

    def save_mask_tiles(self, filename, mask_params, tile_size, level, tx, ty):

        should_skip_tiles = {}

        for image_params in mask_params["images"]:

            output_file = str(image_params["out_path"] / filename)
            path_exists = output_file.exits() or get_empty_path(output_file).exists()
            should_skip = path_exists and image_params.get("is_up_to_date", False)
            should_skip_tiles[output_file] = should_skip

        if all(should_skip_tiles.values()):
            logger.warning(f"Not saving tile level {level} ty {ty} tx {tx}")
            logger.warning(f"Every mask {filename} exists with same rendering settings")
            return should_skip_tiles

        return should_skip_tiles

    def return_tile(self, output_file, settings, tile_size, level, tx, ty):
        raise NotImplementedError

    def save_tile(self, output_file, settings, tile_size, level, tx, ty):
        img = self.return_tile(output_file, settings, tile_size, level, tx, ty)
        img.save(output_file, quality=85)


class MaskTile:
    def __init__(self, img: Optional[Image.Image], file: Path):
        self.img = img
        self.file = file


class TiffOpener(Opener):
    """Handle the access to TIFF File image data."""

    def __init__(self, path):
        super(TiffOpener, self).__init__(path)

    def _setup(self):
        reader = "tifffile"
        io = TiffFile(self.path)
        self.ome_version = self._get_ome_version(io)
        if self.ome_version == 5:
            io = TiffFile(self.path, is_ome=False)
        self.group = zarr.open(self.io.series[0].aszarr())
        # Treat non-pyramids as groups of one array
        if isinstance(self.group, zarr.core.Array):
            root = zarr.group()
            root[0] = self.group
            self.group = root
        logger.info("OME ", self.ome_version)
        num_channels = self.get_shape()[0]
        dimensions = io.series[0].get_axes()
        self.wrapper = ZarrWrapper(self.group, dimensions)

        default_dtype = np.uint16
        tile_0 = self.get_tifffile_tile(0, 0, 0, 0, 1024)
        if tile_0 is not None:
            default_dtype = tile_0.dtype

        if num_channels == 3 and tile_0.dtype == "uint8":
            rgba = True
            rgba_type = "3 channel"
        elif num_channels == 1 and tile_0.dtype == "uint8":
            rgba = True
            rgba_type = "1 channel"
        else:
            rgba = False
            rgba_type = None
        return io, reader, rgba, rgba_type, default_dtype

    @staticmethod
    def _get_ome_version(io):
        try:
            software = io.pages[0].tags[305].value
            sub_ifds = io.pages[0].tags[330].value
            if "Faas" in software or sub_ifds is None:
                return 5
            else:
                return 6
        except Exception as e:
            logger.error("Failed to get OME version.")
            logger.exception(e)
            return 5

    def get_tifffile_tile(self, level, tx, ty, channel_number, tilesize=1024):

        tile = self.read_tiles(level, channel_number, tx, ty, tilesize)

        if tile is None:
            return np.zeros((tilesize, tilesize), dtype=self.default_dtype)

        return tile

    def read_tiles(self, level, channel_number, tx, ty, tilesize):
        ix = tx * tilesize
        iy = ty * tilesize

        try:
            tile = self.wrapper[
                level, ix : ix + tilesize, iy : iy + tilesize, 0, channel_number, 0
            ]
            return tile
        except Exception as e:
            logger.error("Failed to load tiles.")
            logger.exception(e)
            return None

    def read_metadata(self):
        try:
            metadata = ome_types.from_tiff(self.path)
        except Exception as e:
            logger.error(f"Failed to read metadata from {self.path}.")
            logger.exception(e)
            return None

        if not metadata or not metadata.images or not metadata.images[0]:
            return None

        return metadata

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

    def get_level_tiles(self, level, tile_size):
        # Negative indexing to support shape len 3 or len 2
        ny = int(np.ceil(self.group[level].shape[-2] / tile_size))
        nx = int(np.ceil(self.group[level].shape[-1] / tile_size))
        return nx, ny

    @staticmethod
    def parse_shape(shape):
        if len(shape) >= 3:
            (num_channels, shape_y, shape_x) = shape[-3:]
        else:
            (shape_y, shape_x) = shape
            num_channels = 1

        return num_channels, shape_x, shape_y

    def get_shape(self):

        (num_channels, shape_x, shape_y) = self.parse_shape(self.group[0].shape)
        all_levels = [self.parse_shape(v.shape) for v in self.group.values()]
        num_levels = len([shape for shape in all_levels if max(shape[1:]) > 512])
        return num_channels, num_levels, shape_x, shape_y

    def get_tile(self, num_channels, level, tx, ty, channel_number, fmt=None):
        if self.is_rgba("3 channel"):
            tile_0 = self.get_tifffile_tile(level, tx, ty, 0, 1024)
            tile_1 = self.get_tifffile_tile(level, tx, ty, 1, 1024)
            tile_2 = self.get_tifffile_tile(level, tx, ty, 2, 1024)
            tile = np.zeros((tile_0.shape[0], tile_0.shape[1], 3), dtype=np.uint8)
            tile[:, :, 0] = tile_0
            tile[:, :, 1] = tile_1
            tile[:, :, 2] = tile_2
            _format = "I;8"
        else:
            tile = self.get_tifffile_tile(level, tx, ty, channel_number, 1024)
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

    def generate_mask_tiles(
        self, filename, mask_params, tile_size, level, tx, ty, should_skip_tiles={}
    ) -> Iterator[MaskTile]:
        tile = self.get_tifffile_tile(level, tx, ty, 0, tile_size)

        for image_params in mask_params["images"]:

            output_file = image_params["out_path"] / filename
            if should_skip_tiles.get(str(output_file), False):
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
                yield MaskTile(None, empty_file)
            else:
                img = Image.frombytes("RGBA", target.T.shape[1:], target.tobytes())
                yield MaskTile(img, output_file)

    def save_mask_tiles(self, filename, mask_params, tile_size, level, tx, ty):
        should_skip_tiles = super(TiffOpener, self).save_mask_tiles(
            filename, mask_params, tile_size, level, tx, ty
        )

        mask_tiles = self.generate_mask_tiles(
            filename, mask_params, tile_size, level, tx, ty, should_skip_tiles
        )

        for mask_tile in mask_tiles:
            if mask_tile.img:
                img_bytes = BytesIO()
                mask_tile.img.save(
                    img_bytes,
                    compress_level=1,
                    format=mask_tile.file.suffix.lower()[1:],
                )
                mask_tile.file.write_bytes(img_bytes.getvalue())
            elif mask_tile.file is not None:
                if not mask_tile.file.exists():
                    mask_tile.file.touch()

        return should_skip_tiles

    def return_tile(self, output_file, settings, tile_size, level, tx, ty):

        if self.is_rgba("3 channel"):

            tile_0 = self.get_tifffile_tile(level, tx, ty, 0, tile_size)
            tile_1 = self.get_tifffile_tile(level, tx, ty, 1, tile_size)
            tile_2 = self.get_tifffile_tile(level, tx, ty, 2, tile_size)
            tile = np.zeros((tile_0.shape[0], tile_0.shape[1], 3), dtype=np.uint8)
            tile[:, :, 0] = tile_0
            tile[:, :, 1] = tile_1
            tile[:, :, 2] = tile_2

            return Image.fromarray(tile, "RGB")

        elif self.is_rgba("1 channel"):

            tile = self.get_tifffile_tile(level, tx, ty, 0, tile_size)

            return Image.fromarray(tile, "RGB")

        else:
            target = None
            for i, (marker, color, start, end) in enumerate(
                zip(
                    settings["Channel Number"],
                    settings["Color"],
                    settings["Low"],
                    settings["High"],
                )
            ):
                tile = self.get_tifffile_tile(level, tx, ty, int(marker), tile_size)

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
            else:
                return


class SvsOpener(Opener):
    """Handle the wrapping and access to image tile data for SVS files."""

    def _setup(self):
        io = OpenSlide(self.path)
        self.dz = DeepZoomGenerator(io, tile_size=1024, overlap=0, limit_bounds=True)
        reader = "openslide"
        rgba = True
        rgba_type = None
        default_dtype = np.uint8
        return io, reader, rgba, rgba_type, default_dtype

    def get_level_tiles(self, level, tile_size):
        reverse_level = self.dz.level_count - 1 - level
        return self.dz.level_tiles[reverse_level]

    def get_shape(self):

        (width, height) = self.io.dimensions

        def has_one_tile(counts):
            return max(counts) == 1

        small_levels = list(filter(has_one_tile, self.dz.level_tiles))
        level_count = self.dz.level_count - len(small_levels) + 1

        return 3, level_count, width, height

    def get_tile(self, num_channels, level, tx, ty, channel_number, fmt=None):
        reverse_level = self.dz.level_count - 1 - level
        img = self.dz.get_tile(reverse_level, (tx, ty))
        return img

    def return_tile(self, output_file, settings, tile_size, level, tx, ty):
        reverse_level = self.dz.level_count - 1 - level
        return self.dz.get_tile(reverse_level, (tx, ty))
