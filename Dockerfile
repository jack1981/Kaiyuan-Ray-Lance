# syntax=docker/dockerfile:1.7
FROM python:3.11.8-slim-bookworm AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore

RUN set -eux; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends build-essential python3-dev; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install --no-compile --prefix=/install -r /tmp/requirements.txt

FROM python:3.11.8-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_ROOT_USER_ACTION=ignore \
    RAY_USAGE_STATS_ENABLED=0

RUN set -eux; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      tini; \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd --create-home --shell /bin/bash app
WORKDIR /workspace
COPY --chown=app:app . /workspace
RUN install -m 0755 -o app -g app /workspace/script/ray_entrypoint.sh /opt/entrypoint.sh
USER app

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/entrypoint.sh"]
CMD ["bash", "-lc", "if [ -n \"$PIPELINE_CONFIG\" ]; then bash script/run_local.sh main.py \"$PIPELINE_CONFIG\"; else echo 'Usage: docker compose run --rm -e PIPELINE_CONFIG=example/read_write.yaml app'; fi"]
