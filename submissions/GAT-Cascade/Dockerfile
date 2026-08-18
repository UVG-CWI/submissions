FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6"

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    cmake \
    git \
    libopenblas-dev \
    ninja-build \
    python3.8 \
    python3.8-dev \
    python3-pip \
    python3-setuptools \
    python3-wheel \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.8 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && python -m pip install --upgrade pip setuptools wheel

COPY requirements.txt /workspace/requirements.txt

RUN python -m pip install \
    torch==1.7.1+cu110 \
    torchvision==0.8.2+cu110 \
    -f https://download.pytorch.org/whl/torch_stable.html \
    && python -m pip install -r /workspace/requirements.txt

COPY . /workspace

RUN if [ -d /workspace/external/PointCloudUpsampling/Installation_files/MinkowskiEngine ]; then \
      cd /workspace/external/PointCloudUpsampling/Installation_files/MinkowskiEngine && \
      python setup.py install --blas=openblas; \
    else \
      python -m pip install "MinkowskiEngine==0.4.3" --install-option="--blas=openblas"; \
    fi

ENV PYTHON_BIN=python
ENV SPLIT=test

ENTRYPOINT ["bash", "run.sh"]
