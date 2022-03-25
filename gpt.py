import re

import openai
import pandas as pd


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
    features = """\
doll: on the bed, 1 foot tall.
pitcher: on the dresser, round, 1 foot tall.
lamp: on the dresser, 2-3 feet tall.
picture frame: rectangular, on the wall, 2-3 feet tall, 2-3 feet across.
bowl: on the dresser, round, less than 1 foot tall.
bench: on the floor, rectangular, 5 feet long, 1 foot tall, flat surface.
pillow: on the bed, 2-3 feet across.
chair: on the floor, 4 feet tall.
nightstand: next to the bed, 2-3 feet tall, flat surface.
door: 6 feet tall, rectangular.
curtain: 6 feet tall, rectangular, on the wall.
dresser: on the floor, against the wall, 4 feet tall, flat surface.
bed: against the wall, on the floor, 5 feet across, 6 feet long.\
"""
    features = features.split("\n")
    features = dict([f.split(": ") for f in features])
    for name, _ in features.items():
        prompt = "\n".join(f"{k}: {v}" for k, v in features.items() if k != name)
        prompt = f"{prompt}\n{name}:"
        breakpoint()

        response = openai.Completion.create(
            engine="text-davinci-001", prompt=prompt, max_tokens=50
        )
        for choice in response.choices:
            print(f"{name}:", choice.text)


if __name__ == "__main__":
    main()
