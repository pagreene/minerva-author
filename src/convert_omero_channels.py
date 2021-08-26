import argparse
import json
import os
from json.decoder import JSONDecodeError

from util import Path


def make_channel(ch_key, ch):
    ch_id = 0
    try:
        # Minerva Author uses a zero-based index
        ch_id = int(ch_key) - 1
    except ValueError as e:
        print(e)
        print(f'Skipping channel: "{ch_key}" is not an integer')
        return {}
    return {
        "label": ch["label"],
        "color": ch["color"].lower(),
        "min": ch["start"] / ch["max"],
        "max": ch["end"] / ch["max"],
        "id": ch_id,
    }


def make_group(channels):
    all_channels = [make_channel(ch_key, ch) for (ch_key, ch) in channels.items()]
    all_channels = [ch for ch in all_channels if "id" in ch]
    all_channels.sort(key=lambda ch: ch["id"])
    return {"channels": all_channels, "label": "imported omero channels"}


def main(omero_json, author_json):

    story_json_checks = [
        (author_json, [".json"]),
        (Path(author_json.stem), [".groups", ".story"]),
    ]
    test_suffix = lambda check: check[0].suffix in check[1]
    if not all(map(test_suffix, story_json_checks)):
        print(
            " ".join(
                ["Invalid output path.", "It must end in .story.json or .groups.json"]
            )
        )
        return

    if author_json.exists():
        print(f"Overwriting existing save file {author_json}")
    else:
        print(f"Writing to new save file {author_json}")

    if not author_json.parent.exists():
        author_json.parent.mkdir(parents=True)

    omero_channels = {}
    try:
        with omero_json.open() as rf:
            loaded = json.load(rf)
        omero_channels = loaded["channels"]
    except (FileNotFoundError, JSONDecodeError, KeyError):
        print(f"Invalid input file: cannot parse {omero_json}")
        return

    with author_json.open("w") as wf:
        json.dump(
            {
                "in_file": "",
                "csv_file": "",
                "groups": [make_group(omero_channels)],
                "masks": [],
                "waypoints": [],
                "sample_info": {"rotation": 0, "name": "", "text": ""},
            },
            wf,
        )

        print(f"Success! {author_json} written")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "omero_json",
        metavar="omero_json",
        type=Path,
        help="Input path to exported omero channels",
    )
    parser.add_argument(
        "author_json",
        metavar="author_json",
        type=Path,
        help="Output Minerva Author save file with channels from omero",
    )
    args = parser.parse_args()

    omero_json = args.omero_json
    author_json = args.author_json

    main(omero_json, author_json)
