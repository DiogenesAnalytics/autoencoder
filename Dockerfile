# jupyter base image
FROM jupyter/scipy-notebook:lab-4.0.0 as cpu-only

# first turn off git safe.directory
RUN git config --global safe.directory '*'

# turn off poetry venv
ENV POETRY_VIRTUALENVS_CREATE=false

# set src target dir
WORKDIR /usr/local/src/autoencoder

# get src
COPY . .

# get poetry in order to install development dependencies
RUN pip install poetry

# config max workers
RUN poetry config installer.max-workers 10

# now install development dependencies
RUN poetry install --with dev -C .

# additional GPU-enabled steps
FROM cpu-only as gpu-enabled

# get mvp gpu cuda libs
RUN poetry install -E gpu-min

# install CUDA tools
RUN mamba install -y -c conda-forge cudatoolkit=11.8.0 && \
    mamba clean --all -f -y && \
    fix-permissions "${CONDA_DIR}" && \
    fix-permissions "/home/${NB_USER}"

# setting up CUDA library link
RUN export CUDNN_PATH=$(dirname \
    $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)")) && \
    ln -s ${CUDNN_PATH} ${CONDA_DIR}/lib/cudnn.ln

# setting dynamic link lib paths
ENV LD_LIBRARY_PATH=${CONDA_DIR}/lib/:${CONDA_DIR}/lib/cudnn.ln/lib

# host NVIDIA driver minimum version metadata
LABEL nvidia.driver.minimum_version="450.80.02"
