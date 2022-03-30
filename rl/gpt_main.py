from my import main


class Args(main.Args):
    pass


class Trainer(main.Trainer):
    pass


if __name__ == "__main__":
    Trainer.main(Args().parse_args())
