import openai
import pandas as pd
import torch
from tqdm import tqdm
from utils import preformat_attributes


def get_attributes():
    df = pd.read_csv("descriptions.csv")
    for column in ["description", "name"]:
        for values in preformat_attributes(zip(df["name"], df[column])).values():
            yield from values
    for i in range(8):
        csv = f"{i}.csv"
        df = pd.read_csv(csv)
        for column in ["ground_truth", "completion", "name"]:
            for values in preformat_attributes(zip(df["name"], df[column])).values():
                yield from values


def main():
    def get_embedding(text, engine="text-similarity-babbage-001"):
        if text.endswith("."):
            text = text[:-1]
        print(text)
        embedding = openai.Embedding.create(input=[text], engine=engine)["data"][0][
            "embedding"
        ]
        return torch.tensor(embedding)

    embeddings = {attr: get_embedding(attr) for attr in tqdm(list(get_attributes()))}
    torch.save(embeddings, "embeddings.pt")


if __name__ == "__main__":
    main()
