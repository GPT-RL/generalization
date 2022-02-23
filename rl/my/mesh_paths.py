from pathlib import Path
from typing import NamedTuple, Optional, Set

import gym_miniworld
import pandas as pd
from my.env import EXCLUDED, PATH, Mesh


class ObjPng(NamedTuple):
    obj: str
    png: Optional[str]


def mesh_path_to_obj_png(path: Path) -> ObjPng:
    return ObjPng(obj=str(path), png=str(path.with_name("texture_map.png")))


def get_meshes(data_path: Path, names: Optional[str]):

    # default meshes (i.e. those that come with gym-miniworld package)
    default_meshes_dir = Path(Path(gym_miniworld.__file__).parent, "meshes")

    def get_default_meshes():
        for name in {m.stem for m in default_meshes_dir.iterdir()}:
            name = name.replace("_", " ").lower()
            yield Mesh(height=1, mission=[name], name=name, obj=name, png=None)

    default_meshes = list(get_default_meshes())

    # data-path meshes (i.e. the ycb objects)
    def get_data_path_meshes():
        if data_path:
            ycb = pd.read_csv("ycb.csv")
            ycb = ycb[~ycb[EXCLUDED]]

            for _, row in ycb.iterrows():
                path = Path(data_path, row[PATH])
                obj, png = mesh_path_to_obj_png(path)
                name = row["name"].lower()
                height = row.get("height")
                height = 1 if pd.isna(height) else height / 100
                yield Mesh(obj=obj, png=png, mission=[name], name=name, height=height)

    if names is None:
        meshes = default_meshes if data_path is None else list(get_data_path_meshes())
    else:
        names: Set[str] = {n.lower() for n in names.split(",")}
        data_path_meshes = {m.mission: m for m in (list(get_data_path_meshes()))}
        default_meshes = {m.mission: m for m in default_meshes}

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
