# inspired by https://sourcery.ai/blog/python-docker/
FROM nvidia/cudagl:11.4.0-devel-ubuntu20.04 as base
ENV LC_ALL C.UTF-8

# no .pyc files
ENV PYTHONDONTWRITEBYTECODE 1

# traceback on segfau8t
ENV PYTHONFAULTHANDLER 1

# use ipdb for breakpoints
ENV PYTHONBREAKPOINT=ipdb.set_trace

# common dependencies
RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \
      # needed for rendering
      freeglut3-dev \
      libfreetype6 \

      # needed for egl
      libegl-mesa0 \

      # git-state
      git \

      # primary interpreter
      python3.8 \

      # required by transformers package
      python3.8-distutils \

 && apt-get clean

FROM base AS python-deps

# build dependencies
RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \

      # required by poetry
      python3-pip \

      # required for redis
      gcc \

      # required for habitat-sim
      cmake \
      libjpeg-dev \
      libglm-dev \
      libgl1-mesa-glx \
      libegl1-mesa-dev \
      mesa-utils \
      xorg-dev freeglut3-dev \

 && apt-get clean

WORKDIR "/deps"

COPY pyproject.toml poetry.lock /deps/
RUN pip3 install poetry && poetry install

ENV VIRTUAL_ENV=/root/.cache/pypoetry/virtualenvs/generalization-K3BlsyQa-py3.8/

RUN git clone --branch stable https://github.com/facebookresearch/habitat-sim.git \
 && cd habitat-sim \
 && $VIRTUAL_ENV/bin/python setup.py install --headless --with-cuda

FROM base AS runtime


RUN apt-get update -q \
 && DEBIAN_FRONTEND="noninteractive" \
    apt-get install -yq \

      # for habitat-sim examples.py
      wget \

 && apt-get clean \


WORKDIR "/project"
ENV VIRTUAL_ENV=/root/.cache/pypoetry/virtualenvs/generalization-K3BlsyQa-py3.8/
ENV PATH="$VIRTUAL_ENV/bin:$PATH"
COPY --from=python-deps $VIRTUAL_ENV $VIRTUAL_ENV
COPY --from=python-deps /deps/habitat-sim /deps/habitat-sim
COPY main.py .
COPY objectnav_mp3d.yaml .

ENTRYPOINT ["python"]
