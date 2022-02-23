import argparse
from pathlib import Path

from PIL import Image
from transformers import CLIPModel, CLIPProcessor

parser = argparse.ArgumentParser()
parser.add_argument("path", type=Path)
parser.add_argument("--choices", nargs="+")
args = parser.parse_args()
model: CLIPModel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

image = Image.open(str(args.path)).convert("RGB")

inputs = processor(
    # text=args.choices,
    images=image,
    return_tensors="pt",
    padding=True,
)

outputs = model(**inputs)
logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
probs = logits_per_image.softmax(dim=1).reshape(
    -1
)  # we can take the softmax to get the label probabilities
for choice, prob in zip(args.choices, probs):
    print(f"{choice}: {(100 * prob).round()}%")
