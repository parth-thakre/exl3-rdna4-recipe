# ExLlamaV3 with this recipe's RDNA4 patches + TabbyAPI, for the RX 9070 / 9070 XT (gfx1201). Serves Qwen3.8-27B with
# the DFlash2 draft at 96k context (128k with CTX=128k); the models are mounted, not baked in (container/README.md).
#
# The build stage runs the recipe's own setup scripts (setup/) at RECIPE_COMMIT on AMD's ROCm dev image, so the
# extension is built exactly as the README's install builds it. The runtime stage needs no ROCm: the extension
# resolves its ROCm libraries (libamdhip64.so.7, libhipblas.so.3, ...) to the copies the PyTorch ROCm wheel loads
# first, which carry the same sonames. It keeps gcc and the Python headers because Triton compiles a small C helper
# at runtime.
#
#   podman build -f container/Dockerfile --build-arg RECIPE_COMMIT=<commit> -t exl3-rdna4 container   # from GitHub
#   podman build --security-opt label=disable -v "$PWD:/src:ro" --build-arg RECIPE_URL=/src \
#       --build-arg RECIPE_COMMIT=$(git rev-parse HEAD) -f container/Dockerfile -t exl3-rdna4 container   # this checkout
#       (committed state only: the build clones it)
# Base images are pinned by digest: rocm/dev-ubuntu-24.04:7.2.4 and ubuntu:24.04.
FROM docker.io/rocm/dev-ubuntu-24.04@sha256:bdc8e61026cbb844ede93d44d2c50055f51ebb2041906b60182bf3bee3139054 AS build
ARG RECIPE_URL=https://github.com/parth-thakre/exl3-rdna4-recipe
ARG RECIPE_COMMIT=main
ARG MAX_JOBS=4
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc g++ python3.12-venv python3.12-dev git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*
RUN git clone "$RECIPE_URL" /opt/recipe && git -C /opt/recipe -c advice.detachedHead=false checkout "$RECIPE_COMMIT"
WORKDIR /opt/recipe
# Only the ROCm -dev packages the build needs (setup/fetch_deps.sh lists them), not all of rocm-hip-sdk
RUN apt-get update \
    && pkgs=$(NO_FETCH=1 setup/fetch_deps.sh | sed -n 's/^  sudo apt install //p') \
    && { [ -z "$pkgs" ] || apt-get install -y --no-install-recommends $pkgs; } \
    && rm -rf /var/lib/apt/lists/*
RUN setup/fetch_deps.sh \
    && setup/make_venv.sh \
    && MAX_JOBS=$MAX_JOBS setup/build_exllamav3.sh \
    && setup/install_tabby.sh \
    && cp container/config.yml tabbyAPI/config.yml \
    && rm -rf exllamav3/build logs /root/.cache \
    && git rev-parse HEAD > container/RECIPE_COMMIT

FROM docker.io/library/ubuntu@sha256:f610ab94648195aa356059f5b41d6085c9d4d903c072430cdd1af7bdb646106b
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-dev gcc libc6-dev ca-certificates \
    && rm -rf /var/lib/apt/lists/*
COPY --from=build /opt/recipe /opt/recipe
WORKDIR /opt/recipe
# Any uid can run it (the plugin may not run it as root): the log directories and Triton's kernel cache must be
# writable. TabbyAPI writes logs/{time}.log relative to its own working directory (tabbyAPI/), not /opt/recipe/logs,
# so that directory has to exist too - the API server exits at startup without it.
RUN mkdir -p logs tabbyAPI/logs && chmod 1777 logs tabbyAPI/logs
ENV TRITON_CACHE_DIR=/tmp/triton HOME=/tmp
# Models: /models, read-only is fine (container/README.md)
ENV EXL3_MODELS=/models
EXPOSE 8096
ENTRYPOINT ["/opt/recipe/container/entrypoint.sh"]
