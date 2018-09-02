FROM nvidia/cuda:9.1-base-ubuntu16.04

RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*
RUN useradd --create-home --shell /bin/bash containeruser
USER containeruser
WORKDIR /home/containeruser

# install miniconda
RUN curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
    chmod +x ~/miniconda.sh && \
    ~/miniconda.sh -b -p /home/containeruser/conda && \
    rm ~/miniconda.sh && \
    /home/containeruser/conda/bin/conda clean -ya
ENV PATH /home/containeruser/conda/bin:$PATH

# install conda packages
RUN conda install -y \
    python=3.5 && \
    conda install -y -c menpo menpo && \
    conda install -c menpo opencv3 && \
    conda clean -ya

# install pip packages
RUN pip install --upgrade pip && \
    pip install cython \
    opencv-python \
    torch=0.4.1 \
    torchvision \
    tqdm \
    scikit-image \
    scikit-learn \
    tensorboardX \
    pyyaml

ENV PYTHONPATH /home/containeruser/PyTorch:$PYTHONPATH
CMD ["/bin/bash"]