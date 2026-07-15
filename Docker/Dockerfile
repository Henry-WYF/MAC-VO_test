FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# Env vars for the nvidia-container-runtime.
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES graphics,utility,compute

RUN apt-get update
RUN apt-get install -y unzip sudo git wget python3-pip python3-dev \
    build-essential cmake pybind11-dev libboost-serialization-dev \
    libopencv-dev python3-opencv
RUN apt-get install -y ffmpeg libsm6 libxext6 libgtk-3-dev libxkbcommon-x11-0 vulkan-tools

RUN apt-get update
RUN apt-get install -y wget && rm -rf /var/lib/apt/lists/*

ARG USERNAME=macvo
RUN useradd -ms /bin/bash ${USERNAME} 
RUN echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers
# RUN wget \
#     https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh \
#     && bash Miniconda3-latest-Linux-x86_64.sh -b -p /home/${USERNAME}/.conda \
#     && rm -f Miniconda3-latest-Linux-x86_64.sh 
# RUN chown -R ${USERNAME}:${USERNAME} /home/${USERNAME}/.conda
# RUN conda init && . ~/.bashrc

USER root

USER ${USERNAME}
ENV PATH="/home/${USERNAME}/.conda/bin:${PATH}"
WORKDIR /home/${USERNAME}

# Setup environment
SHELL ["/bin/bash", "-l", "-c"]

RUN sudo pip3 install --upgrade pip
RUN sudo pip3 install --no-cache-dir pypose>=0.6.8  
RUN sudo pip3 install --no-cache-dir evo
RUN sudo pip3 install --no-cache-dir matplotlib tabulate tqdm rich cupy-cuda12x einops 
RUN sudo pip3 install --no-cache-dir timm==0.9.12 rerun-sdk==0.21.0 yacs 
RUN sudo pip3 install --no-cache-dir numpy 
RUN sudo pip3 install --no-cache-dir pyyaml wandb pillow scipy flow_vis h5py 
RUN sudo pip3 install --no-cache-dir xformers==0.0.27.post2 onnx
RUN sudo pip3 install --no-cache-dir torchvision jaxtyping typeguard==2.13.3

# Compile the pinned ORB-SLAM3 DBoW2 subset once into the runtime image.  The
# Python and C++ sides both use Ubuntu's system OpenCV to avoid loading a pip
# wheel's private OpenCV alongside libopencv-dev.
USER root
COPY Thirdparty/DBoW2 /tmp/macvo-dbow2
RUN cmake -S /tmp/macvo-dbow2 -B /tmp/macvo-dbow2-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_LIBRARY_OUTPUT_DIRECTORY=/opt/macvo-native \
    && cmake --build /tmp/macvo-dbow2-build --parallel \
    && mkdir -p /usr/share/doc/macvo-dbow2 \
    && cp /tmp/macvo-dbow2/LICENSE.txt /tmp/macvo-dbow2/NOTICE.md \
          /tmp/macvo-dbow2/README.ORB_SLAM3.txt /usr/share/doc/macvo-dbow2/ \
    && rm -rf /tmp/macvo-dbow2-build /tmp/macvo-dbow2
ENV PYTHONPATH="/opt/macvo-native:${PYTHONPATH}"
RUN python3 -c "import cv2, macvo_dbow2; print(cv2.__file__); print(macvo_dbow2.__file__)"
RUN ! pip3 list --format=freeze | grep -Eqi '^opencv-(python|python-headless|contrib-python)='
USER ${USERNAME}

CMD /bin/bash
