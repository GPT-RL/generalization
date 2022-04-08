import random

import openai
import pandas as pd
from tqdm import tqdm


def get_completions():
    features = pd.read_csv("habitat.csv").groupby("id")

    def get_prompts():
        for _, group in features:
            for _, row in group.iterrows():
                prompt = (
                    f"The {row['name']} is{'' if row['room'].startswith('between') else ' in the'} "
                    f"{row['room']}{'' if pd.isna(row['where']) else ', ' + row['where']}. ".lower().capitalize()
                    + f"It is {'' if pd.isna(row['dimensions']) else row['dimensions'] + ', '}at "
                    f"{row['vertical position']}.".lower().capitalize()
                )
                yield row["name"], prompt

    prompts = list(get_prompts())
    for name, ground_truth in tqdm(prompts):
        prompt = [v for k, v in prompts if k != name]
        random.shuffle(prompt)
        prompt = prompt[:70]

        prompt = "\n".join(prompt) + f"\nThe {name} is"
        response = openai.Completion.create(
            engine="text-davinci-001", prompt=prompt, max_tokens=50
        )
        choice, *_ = response.choices

        prefix = f"The {name.lower()} is"
        if not ground_truth.startswith(prefix):
            breakpoint()
        ground_truth = ground_truth[len(prefix) :]

        yield dict(
            name=name, prompt=prompt, completion=choice.text, ground_truth=ground_truth
        )


def main():
    pd.DataFrame.from_records(get_completions()).to_csv("gpt.csv", index=False)


if __name__ == "__main__":
    main()
