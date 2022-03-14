import openai
import pandas as pd
import torch
from tqdm import tqdm


def main():
    ground_truth = pd.read_csv("ycb.csv")
    generated = pd.read_csv("ycb-gpt.csv")
    new = generated["completion"].str.split(" and ", expand=True)
    generated["color"] = new[0]
    generated["shape"] = new[1]
    df = pd.concat([ground_truth, generated], axis=0)
    words = pd.concat(
        [df["name"], df["gpt name"], df["color"], df["shape"]], axis=0
    ).dropna()
    words = words.str.strip().unique()

    def get_embedding(text, engine="text-similarity-babbage-001"):
        text = text.replace("\n", " ")
        embedding = openai.Embedding.create(input=[text], engine=engine)["data"][0][
            "embedding"
        ]
        return torch.tensor(embedding)

    embeddings = {word: get_embedding(word) for word in tqdm(words)}
    torch.save(embeddings, "embeddings.pt")


if __name__ == "__main__":
    main()
