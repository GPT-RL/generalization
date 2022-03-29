#! /usr/bin/env bash
name=issue
docker build -t "$name" .
docker run --rm -it \
  --gpus all \
	-v "$HOME/.cache/data/:/root/.cache/data" \
	"$name" "${@:1}"
