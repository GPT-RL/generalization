from gpt_agent import Agent
from my import main


class Args(main.Args):
    randomize_parameters: bool = False


class Trainer(main.Trainer):
    @classmethod
    def _make_agent(cls, *_args, args: Args, agent_class=Agent, **kwargs):
        return super()._make_agent(
            *_args,
            agent_class=Agent,
            args=args,
            randomize_parameters=args.randomize_parameters,
            **kwargs,
        )


if __name__ == "__main__":
    Trainer.main(Args().parse_args())
