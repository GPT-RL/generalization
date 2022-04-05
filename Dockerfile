# Unfortunately, the most reliable version of habitat-sim is available through conda.
# However, we use the techniques described here:
# (https://pythonspeed.com/articles/conda-docker-image-size/)
# to minimize the impact of conda on our build. The goal here is to minimize image size
# and re-implementation of build logic from habitat-sim and conda
FROM docker.io/continuumio/miniconda3:4.9.2 AS build

# Install the package as normal:
RUN conda create -n habitat

# Install conda-pack (per https://pythonspeed.com/articles/conda-docker-image-size/)
# and habitat-sim
RUN conda install \
  conda-pack \
  habitat-sim==0.2.1 \
  # for running habitat-sim headless:
  headless==1.0=0 \
  # required by habitat-sim and habitat-lab 🙄:
  #pytorch==1.7.0 \
  #torchvision==0.8.0\
  #cudatoolkit==10.2.89 \
  -c pytorch -c conda-forge -c aihabitat

#RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html

# Use conda-pack to create a standalone enviornment
# in /venv:
RUN conda-pack -n habitat -o /tmp/env.tar && \
  mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
  rm /tmp/env.tar

# We've put venv in same path it'll be in final image,
# so now fix up paths:
RUN /venv/bin/conda-unpack

# The runtime-stage image; we can use Debian as the
# base image since the Conda env also includes Python
# for us.
FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04 as base

# Copy /venv from the previous stage:
COPY --from=build /venv /venv
COPY --from=build /opt/conda/ /opt/conda/

RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \
      git \
      redis \

      #libcudnn \
      #libcudnn7 \
      # gym[atari]
      #cmake \
      #zlib1g-dev \

      # cv2
      ffmpeg \
      libsm6 \
      libxext6 \
 && apt-get clean

# add /venv to Path for access to python and pip
ENV PATH="/venv/bin:/opt/conda/bin/:$PATH"
ENV PYTHONBREAKPOINT=ipdb.set_trace

#RUN pip uninstall poetry
 #install remaining deps from poetry
COPY pyproject.toml poetry.lock .
RUN pip install poetry==1.1.12 \
    # https://github.com/python-poetry/poetry/discussions/1879#discussioncomment-216870
    && poetry export --without-hashes -f requirements.txt | pip install -r /dev/stdin 
    # install jaxlib (poetry does not support -f)
    #&& pip install \
      #-U jaxlib==0.1.60+cuda102 \
      #-f https://storage.googleapis.com/jax-releases/jax_releases.html \
    #&& pip install numpy==1.21.1 \
    #&& pip install gym-minigrid==1.0.2

# additional requirements for habitat-lab:
#RUN pip install tensorboard lmdb>=0.98 webdataset==0.1.40 ifcfg ipdb
#RUN git clone --branch pull-request https://github.com/ethanabrooks/habitat-lab.git
#RUN git clone --branch stable https://github.com/facebookresearch/habitat-lab.git
#RUN /bin/bash -c "cd habitat-lab; git checkout d6ed1c0a0e786f16f261de2beafe347f4186d0d8; pip3 install -e ."

WORKDIR "/project"

COPY . .

ENTRYPOINT ["python"]
