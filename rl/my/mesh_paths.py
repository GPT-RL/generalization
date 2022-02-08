import re
from pathlib import Path
from typing import NamedTuple, Optional, Set

import gym_miniworld
import pandas as pd
from my.env import EXCLUDED, PATH, Mesh


def get_original_ycb_meshes(
    data_path: Path,
    obj_pattern: str,
    png_pattern: str,
):
    if data_path:
        if not data_path.exists():
            raise RuntimeError(
                f"""\
        {data_path} does not exist.
        Download dataset using https://github.com/sea-bass/ycb-tools
        """
            )

        def get_names(path: Path):
            name = path.parent.parent.name
            name = re.sub(r"\d+(-[a-z])?_", "", name)
            return name.replace("_", " ")

        objs = {get_names(path): path for path in data_path.glob(obj_pattern)}
        pngs = {get_names(path): path for path in data_path.glob(png_pattern)}
        for n in objs:
            yield Mesh(objs.get(n), pngs.get(n), n)


class ObjPng(NamedTuple):
    obj: str
    png: Optional[str]


def mesh_path_to_obj_png(path: Path) -> ObjPng:
    return ObjPng(obj=str(path), png=str(path.with_name("texture_map.png")))


def get_meshes(data_path: Path, names: Optional[str]):

    # default meshes (i.e. those that come with gym-miniworld package)
    default_meshes_dir = Path(Path(gym_miniworld.__file__).parent, "meshes")
    default_meshes = [
        Mesh(height=1, name=name.replace("_", " "), obj=name, png=None)
        for name in {m.stem for m in default_meshes_dir.iterdir()}
    ]

    # data-path meshes (i.e. the ycb objects)
    def get_data_path_meshes():
        if data_path:
            ycb = pd.read_csv("ycb.csv")
            ycb = ycb[~ycb[EXCLUDED]]

            for _, row in ycb.iterrows():
                path = Path(data_path, row[PATH])
                obj, png = mesh_path_to_obj_png(path)
                height = row.get("height")
                height = 1 if pd.isna(height) else height / 100
                yield Mesh(obj=obj, png=png, name=row["name"], height=height)

    data_path_meshes = list(get_data_path_meshes())

    if names is None:
        meshes = default_meshes if data_path is None else data_path_meshes
    else:
        names: Set[str] = set(names.split(","))
        data_path_meshes = {m.name: m for m in data_path_meshes}
        default_meshes = {m.name: m for m in default_meshes}

        def _get_meshes():
            for name in names:
                if name in data_path_meshes:
                    yield data_path_meshes[name]
                elif name in default_meshes:
                    yield default_meshes[name]
                else:
                    raise RuntimeError(f"Invalid name: {name}")

        meshes = list(_get_meshes())
    return meshes
