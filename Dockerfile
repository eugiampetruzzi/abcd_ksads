# syntax=docker/dockerfile:1

# Runs the full abcd_ksads harmonization + analysis pipeline (the Makefile) in a
# container. No data is baked in: bind-mount the ABCD base directory to /data at
# run time and set ABCD_70=/data (see README, "Running via Docker / Apptainer").
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim

# `make` drives the pipeline; the slim base image does not ship it.
RUN apt-get update \
    && apt-get install --yes --no-install-recommends make \
    && rm -rf /var/lib/apt/lists/*

# uv: copy packages into the venv (Docker layers span filesystems, so hardlinks
# are unavailable), never fetch a Python, and at run time never mutate the
# pre-synced environment -- the last point is what keeps `uv run` working under
# Apptainer's read-only root filesystem. MPLCONFIGDIR keeps matplotlib's cache
# in a writable location when the container runs as an arbitrary UID.
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0 \
    UV_FROZEN=1 \
    UV_NO_SYNC=1 \
    UV_CACHE_DIR=/tmp/uv-cache \
    MPLCONFIGDIR=/tmp/matplotlib

WORKDIR /app

# 1) Dependencies only, as a layer cached on the lockfile alone.
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project --no-dev

# 2) The project package plus everything the pipeline reads or runs.
COPY README.md ./
COPY src/ src/
COPY codebooks/ codebooks/
COPY harmonize/ harmonize/
COPY figures/ figures/
COPY tables/ tables/
COPY Makefile ./
RUN uv sync --no-dev

# The two syncs above populate UV_CACHE_DIR as root. Reset it to an empty,
# world-writable directory so the container can also run as an arbitrary,
# non-root UID (Docker `--user`, or Apptainer's invoking user) -- with
# UV_NO_SYNC set there are no downloads at run time, uv only needs a writable
# cache location. MPLCONFIGDIR is created the same way for matplotlib.
RUN rm -rf /tmp/uv-cache /tmp/matplotlib \
    && mkdir -m 777 /tmp/uv-cache /tmp/matplotlib

# `docker run <image>` runs the whole pipeline; append a target to run a single
# stage, e.g. `docker run <image> figures`.
ENTRYPOINT ["make"]
CMD ["all"]
