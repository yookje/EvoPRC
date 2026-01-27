# Use the official CUDA image as a parent image
# Using CUDA 11.3 for compatibility with PyTorch 1.12.1
FROM nvidia/cuda:11.3.1-devel-ubuntu20.04

# Set the working directory
WORKDIR /workspace

# Prevent interactive prompts during installation
ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.8, pip and git
RUN apt-get update && \
    apt-get install -y software-properties-common && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt-get update && \
    apt-get install -y python3.8 python3.8-dev python3.8-distutils python3-pip git wget && \
    update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.8 1 && \
    python3 -m pip install --upgrade pip

# Install PyTorch packages with CUDA 11.3 support
RUN pip3 install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113

# Install other Python packages
RUN pip3 install matplotlib==3.5.2 && \
    pip3 install numpy==1.23.3 && \
    pip3 install pandas==1.5.1 && \
    pip3 install pytz==2022.1 && \
    pip3 install tqdm==4.64.1

# Clone SIL repository
RUN git clone https://github.com/yookje/EvoPRC.git

# Set the default command to bash
CMD ["bash"]
