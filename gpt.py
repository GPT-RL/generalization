import re

import numpy as np
import openai
import pandas as pd
from tqdm import tqdm


def get_properties(df: pd.DataFrame):
    for i, row in df.iterrows():
        obj, *properties = tuple(row)

        def callback1(pat):
            return pat.group(1).lower()

        def callback2(pat):
            return " " + callback1(pat)

        name = re.sub("^([A-Z])", callback1, obj)
        name = re.sub("([A-Z])", callback2, name)
        name = name.replace("*", "")

        properties = [p for p in properties if isinstance(p, str)]
        yield name, properties


def path_name(name: str):
    return f"{name.replace(' ', '-')}.pkl"


def main():
    df = pd.read_csv("ycb.csv")
    df["gpt name"] = df["gpt name"].fillna(df.pop("name"))
    df = df.rename(columns={"gpt name": "name"})
    df["name"] = df["name"].apply(lambda n: n.lower())
    df = df.set_index("name")
    columns = ["color", "shape"]
    df = df[[*columns, "excluded"]]
    df = df[~df["excluded"]].drop("excluded", axis=1)

    def process_row(r: pd.Series):
        def gen():
            for column in r:
                if not pd.isna(column):
                    yield from column.split(",")

        return list(gen())

    features = df.apply(process_row, axis=1)
    features = features[features.apply(lambda x: bool(x))]

    features = features.to_dict()
    features = {k.lower(): " and ".join(v) for k, v in features.items() if v}

    rng = np.random.default_rng(seed=0)

    completions = {}

    def article(n: str):
        vowel = "aeiou"
        return "an" if n[0] in vowel else "a"

    for name, description in tqdm(features.items(), total=len(features)):
        series2: pd.Series = pd.Series({k: v for k, v in features.items() if k != name})
        df = df.sample(frac=1, random_state=rng)

        prompt = " ".join(
            [f"Describe {article(n)} {n}: {d}." for n, d in series2.iteritems()]
            + [f"Describe {article(name)} {name}:"]
        )
        response = openai.Completion.create(
            engine="text-davinci-001", prompt=prompt, max_tokens=50
        )
        completion, *_ = response.choices[0].text.split(".")
        print(completion)
        completions[name] = completion

    completions = [{"name": n, "completion": d} for n, d in completions.items()]
    df = pd.DataFrame.from_records(completions, index="name")
    df.to_csv("ycb-gpt.csv")


if __name__ == "__main__":
    main()
