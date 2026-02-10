FROM python:3.11.8-slim-bookworm AS builder

RUN set -eux; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends build-essential python3-dev; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

FROM python:3.11.8-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    RAY_USAGE_STATS_ENABLED=0

RUN set -eux; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      wget \
      tini; \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -ms /bin/bash app
WORKDIR /workspace
COPY . /workspace
RUN cp /workspace/script/ray_entrypoint.sh /opt/entrypoint.sh \
    && chmod 0755 /opt/entrypoint.sh \
    && chown app:app /opt/entrypoint.sh \
    && chown -R app:app /workspace
USER app

ENTRYPOINT ["/usr/bin/tini", "--", "/opt/entrypoint.sh"]
CMD ["bash", "-lc", "if [ -n \"$PIPELINE_CONFIG\" ]; then bash script/run_local.sh main.py \"$PIPELINE_CONFIG\"; else echo 'Usage: docker compose run --rm -e PIPELINE_CONFIG=example/read_write.yaml app'; fi"]
