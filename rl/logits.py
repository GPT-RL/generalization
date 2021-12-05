import random
import re
from typing import Literal

import torch
from babyai_env import PlantAnimalWrapper
from tap import Tap
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPTNeoForCausalLM


class Args(Tap):
    random: bool = False
    lines_per_input: int = None
    seed: int = 0
    model_name: Literal[
        "gpt2",
        "gpt2-medium",
        "gpt2-large",
        "gpt2-xl",
        "bert-base-uncased",
        "bert-large-uncased",
        "EleutherAI/gpt-neo-1.3B",
        "EleutherAI/gpt-neo-2.7B",
    ] = "EleutherAI/gpt-neo-2.7B"  # what size of pretrained GPT to use


def main(args: Args):
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
    if "neo" in args.model_name:
        model = GPTNeoForCausalLM.from_pretrained(args.model_name)
    else:
        model = GPT2LMHeadModel.from_pretrained(args.model_name, is_decoder=True)

    # inputs = tokenizer(sys.argv[1], return_tensors="pt")
    keys = [
        "green animal",
        "orange animal",
        "green food",
        "orange food",
        "white animal",
        "white food",
        "purple food",
        "black animal",
        "red animal",
        "red food",
    ]

    random.seed(args.seed)
    regex = r"^[aeiouAEIOU][A-Za-z0-9_]*"

    def article(word):
        return "an" if re.search(regex, word) else "a"

    def get_inputs(is_random: bool):
        def groups():
            for k in keys:
                choice = random.choice(keys) if is_random else k
                key = getattr(PlantAnimalWrapper, choice.replace(" ", "_"))
                yield [
                    f"{article(v).capitalize()} {v} is {article(k)} {k}."
                    for v in PlantAnimalWrapper.replacements[key]
                ]

        groups = list(zip(*groups()))
        lines = [line for group in groups for line in group]
        if args.lines_per_input is not None:
            lines = lines[: args.lines_per_input]
        return " ".join(lines)

    inputs = tokenizer(get_inputs(args.random), return_tensors="pt")

    outputs = model(**inputs)

    logits = outputs.logits
    # For each position in the sequence, calculate the probability distribution of
    # the next token.
    lps = torch.log_softmax(logits[:, :-1], dim=-1)
    # Extract the log-probability that was assigned to the token that actually appears
    # next.
    indices = inputs.input_ids[:, 1:, None]
    lps = lps.gather(-1, indices)
    # Set log-prob of masked items to 0 (same as setting probs to 1)
    # so they don't affect the calculation below.

    # Calculate the log-probability of each option.
    option_lps = lps.sum(dim=1).squeeze()

    # log_probs = torch.log_softmax(outputs.logits.squeeze(0), dim=-1)
    # input_ids = inputs.input_ids.squeeze(0)
    # log_probs = torch.gather(log_probs, 1, input_ids.unsqueeze(0))
    print("log P:", lps.squeeze(-1).detach().numpy().round(2))
    print("log P:", option_lps.sum().item())


if __name__ == "__main__":
    main(Args().parse_args())
