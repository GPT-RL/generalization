import re

import pandas as pd


def main():
    path = "/Users/ethanbrooks/Downloads/Thor Objects - Sheet1.csv"
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        obj, *properties = tuple(row)

        def callback1(pat):
            return pat.group(1).lower()

        def callback2(pat):
            return " " + callback1(pat)

        name = re.sub("^([A-Z])", callback1, obj)
        name = re.sub("([A-Z])", callback2, name)

        properties = [p for p in properties if isinstance(p, str)]
        if properties:
            print(f"{name}: {', '.join(properties)}.")


if __name__ == "__main__":
    main()
