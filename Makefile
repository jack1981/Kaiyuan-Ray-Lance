.PHONY: build run run-examples up down logs shell spark-version prepare-sample prepare-sample-hf prepare-example prepare-examples clean

build:
	docker compose build app

run:
	@test -d data/sample/pcmind_kaiyuan_2b_sample.lance || (echo "Missing sample lance dataset at data/sample/pcmind_kaiyuan_2b_sample.lance. Run: make prepare-sample"; exit 2)
	docker compose run --rm --build app

up:
	docker compose up --build

down:
	docker compose down --remove-orphans

logs:
	docker compose logs -f --tail=200

shell:
	docker compose run --rm app bash

spark-version:
	docker compose run --rm --build app bash -lc 'if [ -n "$${JAVA_HOME:-}" ] && [ ! -x "$${JAVA_HOME}/bin/java" ]; then unset JAVA_HOME; fi; if [ -z "$${JAVA_HOME:-}" ]; then export JAVA_HOME="$$(dirname "$$(dirname "$$(readlink -f "$$(command -v java)")")")"; fi; spark-submit --version'

prepare-sample:
	docker compose run --rm --build app python script/prepare_local_sample.py --rows $${SAMPLE_ROWS:-200} --source-mode synthetic --hf-timeout-seconds $${HF_TIMEOUT_SECONDS:-30} --max-hf-parquet-mb $${MAX_HF_PARQUET_MB:-256}

prepare-sample-hf:
	docker compose run --rm --build app python script/prepare_local_sample.py --rows $${SAMPLE_ROWS:-200} --source-mode $${SOURCE_MODE:-auto} --hf-timeout-seconds $${HF_TIMEOUT_SECONDS:-30} --max-hf-parquet-mb $${MAX_HF_PARQUET_MB:-256}

prepare-examples: prepare-sample

prepare-example: prepare-examples

run-examples:
	docker compose run --rm --build app bash script/run_all_examples.sh

clean:
	docker compose down --remove-orphans --rmi local
