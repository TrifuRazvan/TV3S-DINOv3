FROM nvidia/cuda:12.1.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Basic build tools and libs
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        bzip2 \
        git \
        wget \
        ffmpeg \
        libglib2.0-0 \
        libgl1 \
        build-essential \
        cmake \
        ninja-build \
        python3 \
        python3-distutils \
        python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Install micromamba
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
    | tar -xvj -C /usr/local/bin --strip-components=1 bin/micromamba

ENV MAMBA_ROOT_PREFIX=/opt/micromamba
ENV PATH=${MAMBA_ROOT_PREFIX}/bin:/usr/local/bin:${PATH}

SHELL ["/bin/bash", "-lc"]

# Create tv3s env
RUN micromamba create -y -n tv3s python=3.10 && \
    micromamba clean --all --yes

# Install PyTorch (CUDA 12.1) into env
RUN micromamba run -n tv3s pip install --upgrade pip && \
    micromamba run -n tv3s pip install \
        torch==2.2.0 \
        torchvision==0.17.0 \
        torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121

# Core Python deps for TV3S        
# Note: mamba-ssm is intentionally omitted here — it is built from source below.
# torch/torchvision/torchaudio are pinned to prevent accidental upgrades by transitive deps.
RUN micromamba run -n tv3s pip install \
        torch==2.2.0 \
        torchvision==0.17.0 \
        torchaudio==2.2.0 \
        --index-url https://download.pytorch.org/whl/cu121 && \
    micromamba run -n tv3s pip install \
        timm==0.4.12 \
        numpy==1.26.4 \
        opencv-python==4.8.1.78 \
        yapf==0.40.1 \
        einops \
        matplotlib \
        ipython \
        psutil \
        fast-pytorch-kmeans \
        mmsegmentation==0.11.0 \
        mmengine==0.10.7 \
        huggingface_hub \
        transformers==4.57.6 \
        ninja

# Temporary clone of TV3S just for building 3rdparty deps
RUN git clone https://github.com/Ashesham/TV3S.git /opt/TV3S_build && \
    cd /opt/TV3S_build && \
    git submodule update --init --recursive || true

# Build MMCV with CUDA ops from the TV3S 3rdparty tree
RUN micromamba run -n tv3s pip install "setuptools==69.5.1" wheel
RUN cd /opt/TV3S_build/3rdparty/mmcv && \
    MMCV_WITH_OPS=1 micromamba run -n tv3s \
        python -m pip install -v -e . --no-build-isolation

# Build causal-conv1d (CUDA extension) — v1.5.0.post8 as per official TV3S guide
RUN cd /opt/TV3S_build/3rdparty && \
    git clone https://github.com/Dao-AILab/causal-conv1d.git -b v1.5.0.post8 && \
    cd causal-conv1d && \
    MAX_JOBS=16 micromamba run -n tv3s python setup.py install

# Build mamba v1.0.0 from source as per official TV3S installation guide
RUN cd /opt/TV3S_build/3rdparty && \
    git clone --branch v1.0.0 https://github.com/state-spaces/mamba.git && \
    cd mamba && \
    micromamba run -n tv3s python setup.py install

# Fix mamba v1.0.0 incompatibility with transformers >= 4.44:
# mamba_ssm/utils/generation.py imports classes removed from transformers.generation.
# TV3S never uses mamba's generation utilities (text gen only), so stubbing is safe.
RUN micromamba run -n tv3s python3 -c "\
path = '/opt/micromamba/envs/tv3s/lib/python3.10/site-packages/mamba_ssm-1.0.0-py3.10-linux-x86_64.egg/mamba_ssm/utils/generation.py';\
content = open(path).read();\
content = content.replace(\
    'from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput',\
    'try:\n    from transformers.generation import GreedySearchDecoderOnlyOutput, SampleDecoderOnlyOutput\nexcept ImportError:\n    GreedySearchDecoderOnlyOutput = None\n    SampleDecoderOnlyOutput = None'\
);\
open(path, 'w').write(content)"

# Fix mamba v1.0.0 + causal-conv1d v1.5.0.post8 incompatibility:
# causal_conv1d_fn added seq_idx/initial_states before activation; patch to use keyword arg.
RUN sed -i 's/                    self\.activation,/                    activation=self.activation,/' \
    /opt/micromamba/envs/tv3s/lib/python3.10/site-packages/mamba_ssm-1.0.0-py3.10-linux-x86_64.egg/mamba_ssm/modules/mamba_simple.py

# Fix numpy 1.24+ compatibility in mmseg (np.float/int/bool/complex removed)
RUN sed -i 's/np\.float\b/float/g; s/np\.int\b/int/g; s/np\.complex\b/complex/g; s/np\.bool\b/bool/g' \
    /opt/micromamba/envs/tv3s/lib/python3.10/site-packages/mmseg/core/evaluation/metrics.py

# Make tv3s env the default on PATH
ENV PATH=${MAMBA_ROOT_PREFIX}/envs/tv3s/bin:${PATH}

# Ensure PyTorch's shared libs are visible to custom CUDA extensions
ENV LD_LIBRARY_PATH=/opt/micromamba/envs/tv3s/lib/python3.10/site-packages/torch/lib:${LD_LIBRARY_PATH}

# Your actual TV3S repo will be bind-mounted here at runtime
WORKDIR /workspace/TV3S

CMD ["micromamba", "run", "-n", "tv3s", "bash"]
