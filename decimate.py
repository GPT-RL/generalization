from pathlib import Path

import pymeshlab
from PIL import Image
from tap import Tap
from tqdm import tqdm


class Args(Tap):
    input_dir: str
    output_dir: str
    percent: float


def main(args: Args):

    input_dir = Path(args.input_dir).expanduser()
    output_dir = Path(args.output_dir).expanduser()
    output_dir.mkdir(exist_ok=True, parents=True)
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
        ms.save_current_mesh(str(output_path))
        for path in output_path.parent.glob("*.png"):
            # from https://stackoverflow.com/a/273962
            img: Image = Image.open(path)
            w, h = [round(d * args.percent) for d in img.size]

            if "P" in img.mode:  # check if image is a palette type
                img = img.convert("RGB")  # convert it to RGB
                img = img.resize((w, h), Image.ANTIALIAS)  # resize it
                img = img.convert("P", dither=Image.NONE, palette=Image.ADAPTIVE)
                # convert back to palette
            else:
                img = img.resize((w, h), Image.ANTIALIAS)  # regular resize
            img.save(
                output_path.with_name(path.name)
            )  # save the image to the new source
            # img.save(newSourceFile, quality = 95, dpi=(72,72), optimize = True)
            # set quality, dpi , and shrink size


if __name__ == "__main__":
    main(Args().parse_args())
