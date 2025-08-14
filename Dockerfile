# For more information, please refer to https://aka.ms/vscode-docker-python
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/aghcr.io/astral-sh/uv:0.8.6 /uv /uvx /bin/
COPY --from=nvidia/cuda:11.6.1-cudnn8-runtime-ubuntu20.04 /usr/lib/x86_64-linux-gnu/libcudnn.so.8 \
  /usr/lib/x86_64-linux-gnu/libcudnn.so.8.4.0 \
  /usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8 \
  /usr/lib/x86_64-linux-gnu/libcudnn_adv_infer.so.8.4.0 \
  /usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8 \
  /usr/lib/x86_64-linux-gnu/libcudnn_adv_train.so.8.4.0 \
  /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8 \
  /usr/lib/x86_64-linux-gnu/libcudnn_cnn_infer.so.8.4.0 \
  /usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8 \
  /usr/lib/x86_64-linux-gnu/libcudnn_cnn_train.so.8.4.0 \
  /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8 \
  /usr/lib/x86_64-linux-gnu/libcudnn_ops_infer.so.8.4.0 \
  /usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8 \
  /usr/lib/x86_64-linux-gnu/libcudnn_ops_train.so.8.4.0 \
  /usr/lib/x86_64-linux-gnu/

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
    uv sync --locked --no-install-project --no-dev

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Then, add the rest of the project source code and install it
# Installing separately from its dependencies allows optimal layer caching
COPY . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev
ENV PATH="/app/.venv/bin:$PATH"

RUN uv run /app/main.py . --cache cuda

#ENTRYPOINT ["uv", "run", "main.py"]
#ENTRYPOINT [ "deepFilter"]
