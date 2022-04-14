import random

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm


def get_descriptions(test_names, features):
    for _, group in features:
        for _, row in group.iterrows():
            values = {}
            for column in ["room", "where", "dimensions", "vertical position"]:
                if pd.isna(row[column]):
                    values[column] = None
                else:
                    values[column] = row[column].lower()

            room = values["room"]
            where = values["where"]
            dimensions = values["dimensions"]
            vertical_pos = values["vertical position"]

            where_clause = "" if where is None else (", " + where)
            dimensions_clause = "" if dimensions is None else (dimensions + ", ")
            prompt = (
                f"in the {room}{where_clause}. "
                f"It is {dimensions_clause}at "
                f"{vertical_pos}."
            )

            test = row["name"] in test_names
            yield test, row["name"], prompt,


def get_completions(df: pd.DataFrame, i: int) -> list:
    features = df.groupby("id")
    names = df["name"].unique()

    rng = np.random.default_rng(i)
    test_names = rng.choice(names, size=5, replace=False)

    descriptions = list(get_descriptions(test_names, features))
    ground_truths = {k: v for _, k, v in descriptions}
    for name in tqdm(test_names):
        prompts = [f"The {n.lower()} is {p}" for t, n, p, in descriptions if not t]

        random.shuffle(prompts)
        prompts = prompts[:70]

        prompt = "\n".join(prompts) + f"\nThe {name.lower()} is"

        response = openai.Completion.create(
            engine="text-davinci-001", prompt=prompt, max_tokens=50
        )
        choice, *_ = response.choices
        completion, *_ = choice.text.lstrip().split("\n")
        ground_truth = ground_truths[name]

        yield dict(
            name=name,
            prompt=prompt,
            completion=completion,
            ground_truth=ground_truth,
        )


def main():
    df = pd.read_csv("habitat.csv")
    df = df[df["enabled"]]

    for i in range(8):
        pd.DataFrame.from_records(get_completions(df, i)).to_csv(
            f"{i}.csv", index=False
        )

    features = df.groupby("id")
    descriptions = [
        dict(name=name, description=description)
        for _, name, description in get_descriptions([], features)
    ]

    df = pd.DataFrame.from_records(descriptions)
    df.to_csv("descriptions.csv", index=False)


if __name__ == "__main__":
    main()
