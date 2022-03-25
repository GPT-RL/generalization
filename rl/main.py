from typing import cast

from my.main import Args, ArgsType, Trainer

if __name__ == "__main__":
    Trainer.main(cast(ArgsType, Args().parse_args()))
