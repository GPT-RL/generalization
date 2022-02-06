import shutil
from pathlib import Path

import pymeshlab
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    input_dir: str
    output_dir: str
    percent: float


def main(args: Args):

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir()
    objs = list(input_dir.glob("**/*.obj"))
    for input_obj in tqdm(objs):
        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(str(input_obj))
        ms.simplification_quadric_edge_collapse_decimation_with_texture(
            targetperc=args.percent
        )
        output_path = input_obj.relative_to(input_dir)
        output_path = Path(output_dir, output_path)
        output_path.parent.mkdir(exist_ok=True, parents=True)
        for path in input_obj.parent.iterdir():
            if path.name != output_path.name and path.suffix in ("mtl", "obj", "png"):
                shutil.copy2(path, output_path.with_name(path.name))
        ms.save_current_mesh(str(output_path))


if __name__ == "__main__":
    main(Args().parse_args())
