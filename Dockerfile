# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/aghcr.io/astral-sh/uv:0.8.6 /uv /uvx /bin/

RUN apt-get update && apt-get install -y \
  curl \
  build-essential \
  git \
  zlib1g

RUN curl https://sh.rustup.rs -sSf | sh -s -- -y

ENV PATH="/root/.cargo/bin:${PATH}"

# Install the project into `/app`
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy from the cache instead of linking since it's a mounted volume
ENV UV_LINK_MODE=copy

# Install the project's dependencies using the lockfile and settings
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv venv
    #uv sync --locked --no-install-project --no-dev \
RUN  --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install --no-deps --index "https://download.pytorch.org/whl/cu128" \
      "torch==2.7.1+cu128" \
      "torchaudio" \
      "triton" \
      "pyannote.audio==3.3.2" 
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv pip install "aligned-textgrid" \
      "click" \
      "librosa"\
      "torch" \
      "tqdm" \
      "whisperx"\
      "git+https://github.com/JoFrhwld/pympi@pyproject"



# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
# RUN --mount=type=cache,target=/root/.cache/uv \
#     uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"
ENV LD_LIBRARY_PATH="/venv/lib/python3.12/site-packages/nvidia/cudnn/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

#RUN uv run /app/main.py . --cache cuda

#ENTRYPOINT ["uv", "run", "main.py"]
#ENTRYPOINT [ "deepFilter"]
