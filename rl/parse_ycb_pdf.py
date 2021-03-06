import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, cast

import Levenshtein
import pandas as pd
import tabula
from my.env import Mesh
from tap import Tap


@dataclass
class Object:
    dimensions: List[float] = None
    name: str = None
    path: Path = None


def remove_newline(s: str):
    return s.replace("\n", " ")


class Args(Tap):
    data_path: str = str(Path(Path.home(), ".cache/data/ycb"))
    pdf_path: str = "ycb.pdf"


def get_data_path_meshes(data_path: Path, obj_pattern: str, png_pattern: str):
    if data_path:
        data_path = data_path.expanduser()
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


def main(args: Args):
    data_path = Path(args.data_path)
    data_path_meshes = get_data_path_meshes(
        data_path, "*/*/textured.obj", "*/*/texture_map.png"
    )
    mesh_paths = [m.obj.relative_to(data_path) for m in data_path_meshes]
    mesh_paths = {path.parts[0]: path for path in mesh_paths}

    @dataclass
    class Columns:
        Dimensions: int
        Name: int

    renames = {
        0: Columns(Dimensions=9, Name=4),
        1: Columns(Dimensions=3, Name=1),
        3: Columns(Dimensions=2, Name=0),
        5: Columns(Dimensions=4, Name=1),
        7: Columns(Dimensions=2, Name=0),
        8: Columns(Dimensions=4, Name=1),
        10: Columns(Dimensions=3, Name=1),
        12: Columns(Dimensions=4, Name=1),
    }

    hard_coded_paths = {
        "Srub Cleanser bottle": "021_bleach_cleanser/google_16k/textured.obj"
    }

    def generate_objects():
        df: pd.DataFrame
        tables = tabula.read_pdf(
            args.pdf_path, pages="all", pandas_options=dict(header=None)
        )
        for i, df in enumerate(tables):
            if i == 0:
                df = df.drop([0])
            if len(df.columns) == 1:
                continue
            if i not in renames:
                continue
            columns = {v: k for k, v in asdict(renames[i]).items()}
            df = df.rename(columns=columns)

            for _, row in df.iterrows():
                if not pd.isna(row.Name):
                    if pd.isna(row.Dimensions) or row.Dimensions == "N/A":
                        dimensions = []
                    else:
                        dimensions = row.Dimensions.replace("(\t\r  base)", "")
                        dimensions = re.split(r"[~\[\],/\t\r x]+", dimensions)
                        dimensions = [float(d) for d in dimensions if d]
                    name = cast(str, row.Name)
                    name = name.replace("-\xad???", "-")
                    name = re.sub(r"[\t\r ]+", " ", name)
                    try:
                        path = hard_coded_paths[name]
                    except KeyError:
                        path = mesh_paths[
                            min(
                                list(mesh_paths),
                                key=lambda p: Levenshtein.distance(name, p),
                            )
                        ]
                    obj = Object(
                        dimensions=dimensions,
                        name=name,
                        path=path,
                    )
                    yield obj

    dataframe = pd.DataFrame.from_records([asdict(o) for o in generate_objects()])
    dataframe.to_csv("ycb-auto.csv")
    print(dataframe)


if __name__ == "__main__":
    main(Args().parse_args())
