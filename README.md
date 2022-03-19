Download habitat data:
```
wget -P ~/.cache/data https://dl.fbaipublicfiles.com/habitat/data/datasets/objectnav/m3d/v1/objectnav_mp3d_v1.zip
unzip ~/.cache/data/objectnav_mp3d_v1.zip -d ~/.cache/data/
python2 download_mp.py --id HxpKQynjfin --task habitat -o ~/.cache/data/
unzip ~/.cache/data/v1/tasks/mp3d_habitat.zip -d ~/.cache/data/tasks/
```
