import re
from pathlib import Path

import numpy as np
import openai
import pandas as pd
from colors import color
from tap import Tap


class Args(Tap):
    df_dir: str = "/tmp/"
    start_with: str = None


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


def main(args: Args):
    if args.start_with is None:
        df_path = Path(args.df_dir, "thor.pkl")
    else:
        df_path = Path(args.df_dir, path_name(args.start_with))
    if df_path.exists():
        df = pd.read_pickle(str(df_path))
    else:
        csv_path = "/Users/ethanbrooks/Downloads/Thor Objects - Sheet1.csv"
        df = pd.read_csv(csv_path)
        names, properties = zip(*get_properties(df))
        df = pd.DataFrame(dict(names=names, properties=properties))
        df.to_pickle(str(df_path))

    rng = np.random.default_rng(seed=0)
    names = df.names.to_list()
    properties = df.properties.to_list()
    started = args.start_with is None
    for i, name in enumerate(names):
        if not started and name == args.start_with:
            started = True
        if not started:
            continue
        lines = [
            f"{n}: {', '.join(p)}."
            for j, (n, p) in enumerate(zip(names, properties))
            if p and j != i
        ]
        rng.shuffle(lines)
        prompt = f"{' '.join(lines)} {name}:"

        def print_prompt(text):
            print(color(text, style="bold+underline"))

        print_prompt("Enter 1 for:")
        response = openai.Completion.create(
            engine="text-davinci-001", prompt=prompt, max_tokens=50
        )
        completion, *_ = response.choices[0].text.split(".")
        print(f"{name}: {completion}.")
        if properties[i]:
            print_prompt("Enter 2 for:")
            print(f"{name}: {', '.join(properties[i])}.")
        print_prompt("Or enter custom text to replace:")
        entry = input("")
        try:
            entry = int(entry)
            if entry == 1:
                choice = completion.split(", ")
            else:
                assert entry == 2
                choice = None
        except (ValueError, AssertionError):
            choice = entry.replace(".", "").split(", ")
        if choice:
            properties[i] = choice
        else:
            choice = properties[i]
        print(f"Assigned properties to {name}:")
        print(choice)
        new_path = str(df_path.with_name(path_name(name)))
        pd.DataFrame(dict(names=names, properties=properties)).to_pickle(new_path)
        print(f"\nWrote dataframe to {new_path}.")


if __name__ == "__main__":
    main(Args().parse_args())
