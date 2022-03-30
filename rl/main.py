import os
from typing import cast

import pyglet

pyglet.options["headless"] = True

try:
    try:
        visible_devices = os.environ["CUDA_VISIBLE_DEVICES"]
    except KeyError:
        visible_devices = 0
    device = int(visible_devices)
except ValueError:
    device = None
pyglet.options["headless_device"] = device
from base_main import ArgsType  # noqa: E402
from gpt_main import Args, Trainer  # noqa: E402

if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
