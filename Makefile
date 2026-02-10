KIND_CLUSTER_NAME ?= kaiyuan-ray
K8S_NAMESPACE ?= kaiyuan-ray
RAY_K8S_IMAGE ?= kaiyuan-ray-app:latest
PIPELINE_CONFIG ?= example/read_write.yaml
MINIO_BUCKET ?= kaiyuan-ray
MINIO_NODE_ENDPOINT ?= host.docker.internal:30900
RAY_ADDRESS ?= auto
KUBERAY_OPERATOR_KUSTOMIZE ?= github.com/ray-project/kuberay/ray-operator/config/default?ref=v1.4.2

.PHONY: build run run-examples bench up down logs shell prepare-sample prepare-sample-hf prepare-example prepare-examples clean \
	k8s-up k8s-prepare k8s-run k8s-run-examples k8s-logs k8s-down k8s-ui k8s-history k8s-dashboard-port-forward k8s-history-port-forward

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

prepare-sample:
	docker compose run --rm --build app python script/prepare_local_sample.py --rows $${SAMPLE_ROWS:-200} --source-mode synthetic --hf-timeout-seconds $${HF_TIMEOUT_SECONDS:-30} --max-hf-parquet-mb $${MAX_HF_PARQUET_MB:-256}

prepare-sample-hf:
	docker compose run --rm --build app python script/prepare_local_sample.py --rows $${SAMPLE_ROWS:-200} --source-mode $${SOURCE_MODE:-auto} --hf-timeout-seconds $${HF_TIMEOUT_SECONDS:-30} --max-hf-parquet-mb $${MAX_HF_PARQUET_MB:-256}

prepare-examples: prepare-sample

prepare-example: prepare-examples

run-examples:
	docker compose run --rm --build app bash script/run_all_examples.sh

bench:
	docker compose run --rm --build app python script/bench.py --config "$${PIPELINE_CONFIG:-$(PIPELINE_CONFIG)}" --repeat "$${BENCH_REPEAT:-1}" "$${BENCH_EXTRA_ARGS:-}"

k8s-up:
	@command -v kind >/dev/null 2>&1 || (echo "kind is required. Install: https://kind.sigs.k8s.io/"; exit 2)
	@command -v kubectl >/dev/null 2>&1 || (echo "kubectl is required. Install: https://kubernetes.io/docs/tasks/tools/"; exit 2)
	@kind get clusters | grep -qx "$(KIND_CLUSTER_NAME)" || kind create cluster --name "$(KIND_CLUSTER_NAME)" --config k8s/kind/cluster.yaml
	docker compose build app
	kind load docker-image "$(RAY_K8S_IMAGE)" --name "$(KIND_CLUSTER_NAME)"
	kubectl apply --server-side -k "$(KUBERAY_OPERATOR_KUSTOMIZE)"
	kubectl apply -f k8s/base/namespace.yaml
	kubectl apply -f k8s/base/rbac.yaml
	kubectl apply -f k8s/base/minio.yaml
	kubectl -n "$(K8S_NAMESPACE)" rollout status deployment/minio --timeout=180s
	kubectl -n "$(K8S_NAMESPACE)" delete -f k8s/base/minio-init-job.yaml --ignore-not-found
	kubectl apply -f k8s/base/minio-init-job.yaml
	kubectl -n "$(K8S_NAMESPACE)" wait --for=condition=complete job/minio-init --timeout=180s
	kubectl apply -f k8s/kuberay/raycluster.yaml
	kubectl apply -f k8s/kuberay/ray-dashboard-service.yaml
	kubectl apply -f k8s/kuberay/ray-history-server.yaml
	kubectl -n "$(K8S_NAMESPACE)" wait --for=condition=HeadPodReady raycluster/raycluster-kaiyuan --timeout=300s
	kubectl -n "$(K8S_NAMESPACE)" rollout status deployment/ray-history-server --timeout=180s

k8s-prepare: prepare-examples
	docker compose run --rm --build app python script/sync_assets_to_minio.py \
		--endpoint $${MINIO_SYNC_ENDPOINT:-$(MINIO_NODE_ENDPOINT)} \
		--scheme $${MINIO_SCHEME:-http} \
		--bucket $${MINIO_BUCKET:-$(MINIO_BUCKET)} \
		--access-key $${MINIO_ACCESS_KEY:-minio} \
		--secret-key $${MINIO_SECRET_KEY:-minio123}

k8s-run:
	K8S_NAMESPACE=$${K8S_NAMESPACE:-$(K8S_NAMESPACE)} \
	RAY_JOB_IMAGE=$${RAY_JOB_IMAGE:-$(RAY_K8S_IMAGE)} \
	RAY_ADDRESS=$${RAY_ADDRESS:-$(RAY_ADDRESS)} \
	bash script/run_k8s.sh main.py "$${PIPELINE_CONFIG:-$(PIPELINE_CONFIG)}"

k8s-run-examples:
	K8S_NAMESPACE=$${K8S_NAMESPACE:-$(K8S_NAMESPACE)} \
	RAY_JOB_IMAGE=$${RAY_JOB_IMAGE:-$(RAY_K8S_IMAGE)} \
	RAY_ADDRESS=$${RAY_ADDRESS:-$(RAY_ADDRESS)} \
	bash script/run_all_examples_k8s.sh example

k8s-logs:
	kubectl -n "$${K8S_NAMESPACE:-$(K8S_NAMESPACE)}" get pods

k8s-ui:
	@echo "Ray Dashboard UI: http://localhost:30265"
	@echo "Ray Job History UI: http://localhost:30080"

k8s-history:
	kubectl -n "$${K8S_NAMESPACE:-$(K8S_NAMESPACE)}" get rayjobs -o wide

k8s-dashboard-port-forward:
	kubectl -n "$${K8S_NAMESPACE:-$(K8S_NAMESPACE)}" port-forward svc/ray-dashboard 8265:8265

k8s-history-port-forward:
	kubectl -n "$${K8S_NAMESPACE:-$(K8S_NAMESPACE)}" port-forward svc/ray-history-server 8080:8080

k8s-down:
	@if kind get clusters | grep -qx "$(KIND_CLUSTER_NAME)"; then kind delete cluster --name "$(KIND_CLUSTER_NAME)"; else echo "kind cluster '$(KIND_CLUSTER_NAME)' does not exist"; fi

clean:
	docker compose down --remove-orphans --rmi local
