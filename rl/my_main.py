from typing import cast

import pyglet

pyglet.options["headless"] = True
from gpt_main import Args, ArgsType, Trainer  # noqa: E402

if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
