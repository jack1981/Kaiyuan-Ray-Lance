FROM python:3.11.8-slim-bookworm AS builder

RUN set -eux; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends build-essential python3-dev; \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --prefix=/install -r /tmp/requirements.txt

FROM python:3.11.8-slim-bookworm

ARG SPARK_VERSION=3.5.8
ARG LANCE_SPARK_VERSION=0.0.15
ARG LANCE_SPARK_ARTIFACT=lance-spark-bundle-3.5_2.12-${LANCE_SPARK_VERSION}.jar
ARG LANCE_SPARK_BASE_URL=

ENV SPARK_HOME=/opt/spark \
    PATH=/opt/spark/bin:$PATH \
    PYSPARK_PYTHON=python \
    JAVA_HOME=/usr/lib/jvm/default-java \
    LANCE_SPARK_PACKAGE=com.lancedb:lance-spark-bundle-3.5_2.12:0.0.15

RUN set -eux; \
    apt-get -o Acquire::Retries=3 update; \
    apt-get install -y --no-install-recommends \
      ca-certificates \
      curl \
      openjdk-17-jre-headless; \
    java_bin="$(readlink -f "$(command -v java)")"; \
    java_home="$(dirname "$(dirname "${java_bin}")")"; \
    ln -sfn "${java_home}" /usr/lib/jvm/default-java; \
    ln -sfn "${java_home}" /usr/lib/jvm/java-17-openjdk-amd64; \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local

RUN set -eux; \
    pyspark_home="$(python -c 'import pathlib,pyspark; print(pathlib.Path(pyspark.__file__).resolve().parent)')"; \
    ln -sfn "${pyspark_home}" "${SPARK_HOME}"; \
    if [ ! -x "${SPARK_HOME}/bin/spark-submit" ]; then \
      echo "Unable to locate spark-submit under ${SPARK_HOME}" >&2; \
      exit 1; \
    fi; \
    lance_ok=0; \
    for m in \
      "${LANCE_SPARK_BASE_URL}" \
      "https://repo1.maven.org/maven2/com/lancedb/lance-spark-bundle-3.5_2.12/${LANCE_SPARK_VERSION}" \
      "https://repo.maven.apache.org/maven2/com/lancedb/lance-spark-bundle-3.5_2.12/${LANCE_SPARK_VERSION}"; do \
      [ -n "${m}" ] || continue; \
      if curl -fsSL --retry 5 --retry-all-errors --connect-timeout 30 "${m}/${LANCE_SPARK_ARTIFACT}" -o "${SPARK_HOME}/jars/${LANCE_SPARK_ARTIFACT}"; then \
        lance_ok=1; \
        break; \
      fi; \
    done; \
    if [ "${lance_ok}" != "1" ]; then \
      echo "Warning: Unable to pre-download ${LANCE_SPARK_ARTIFACT}; run_local.sh will use --packages fallback at runtime." >&2; \
    fi; \
    spark-submit --version

RUN useradd -ms /bin/bash app
WORKDIR /workspace
COPY . /workspace
RUN chown -R app:app /workspace
USER app

CMD ["bash", "-lc", "if [ -n \"$PIPELINE_CONFIG\" ]; then bash script/run_local.sh main.py \"$PIPELINE_CONFIG\"; else echo 'Usage: docker compose run --rm -e PIPELINE_CONFIG=example/read_write.yaml app'; fi"]
