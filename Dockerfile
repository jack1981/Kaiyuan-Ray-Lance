FROM python:3.11.8-slim-bookworm AS builder

RUN set -eux; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends build-essential python3-dev; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

FROM python:3.11.8-slim-bookworm

ARG SPARK_VERSION=3.5.8
ARG SPARK_PACKAGE=spark-${SPARK_VERSION}-bin-hadoop3
ARG SPARK_BASE_URL=

ENV SPARK_HOME=/opt/spark \
    PATH=/opt/spark/bin:$PATH \
    PYSPARK_PYTHON=python

RUN set -eux; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      openjdk-17-jre-headless; \
    base_url="${SPARK_BASE_URL}"; \
    for u in \
      "${base_url}" \
      "https://downloads.apache.org/spark/spark-${SPARK_VERSION}" \
      "https://dlcdn.apache.org/spark/spark-${SPARK_VERSION}" \
      "https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}"; do \
      [ -n "${u}" ] || continue; \
      if curl -fsSLI --retry 5 --retry-all-errors --connect-timeout 15 "${u}/${SPARK_PACKAGE}.tgz" >/dev/null; then \
        base_url="${u}"; \
        break; \
      fi; \
    done; \
    if [ -z "${base_url}" ]; then \
      echo "Unable to reach Apache Spark mirrors for version ${SPARK_VERSION}" >&2; \
      exit 1; \
    fi; \
    echo "Using Spark mirror: ${base_url}"; \
    curl -fsSL --retry 5 --retry-all-errors --connect-timeout 30 "${base_url}/${SPARK_PACKAGE}.tgz" -o /tmp/${SPARK_PACKAGE}.tgz; \
    curl -fsSL --retry 5 --retry-all-errors --connect-timeout 30 "${base_url}/${SPARK_PACKAGE}.tgz.sha512" -o /tmp/${SPARK_PACKAGE}.tgz.sha512; \
    cd /tmp; \
    sha512sum -c ${SPARK_PACKAGE}.tgz.sha512; \
    tar -xzf ${SPARK_PACKAGE}.tgz -C /opt; \
    ln -s /opt/${SPARK_PACKAGE} ${SPARK_HOME}; \
    rm -f /tmp/${SPARK_PACKAGE}.tgz /tmp/${SPARK_PACKAGE}.tgz.sha512; \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN useradd -ms /bin/bash app
WORKDIR /workspace
COPY . /workspace
RUN chown -R app:app /workspace
USER app

CMD ["bash", "-lc", "if [ -n \"$PIPELINE_CONFIG\" ]; then bash script/run_local.sh main.py \"$PIPELINE_CONFIG\"; else echo 'Usage: docker compose run --rm -e PIPELINE_CONFIG=example/read_write.yaml app'; fi"]
