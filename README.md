# Kaiyuan Spark

[![License](https://img.shields.io/badge/License-Apache-f5de53?&color=f5de53)](LICENSE)
[![arXiv-2512.07612](https://img.shields.io/badge/arXiv-2512.07612-b31b1b.svg?style=flat)](https://arxiv.org/abs/2512.07612)

A scalable data preprocessing framework built on PySpark for [PCMind-2.1-Kaiyuan-2B](https://huggingface.co/thu-pacman/PCMind-2.1-Kaiyuan-2B), a leading fully open-source language model with 2B total parameters (1.4B non-embedding parameters).

## Overview

This framework provides a tree-structured pipeline framework for large-scale data preprocessing, where input files serve as leaf nodes and the final output as the root. The design emphasizes:

- **User-friendly**: Declarative YAML-based configuration
- **Scalable**: Built on PySpark for distributed processing
- **Modular**: Extensible pipeline nodes for custom operations
- **Production-ready**: Supports both local development and YARN cluster deployment

## Quick Start

### Environment Setup

Please [deploy Spark](https://spark.apache.org/docs/latest/quick-start.html) on your server first. Then install the Python dependencies:

```bash
pip install -r requirements.txt
```

### Run Example Pipeline

Modify the paths accordingly in `example/read_write.yaml` and execute:

```bash
bash script/run_yarn.sh main.py example/read_write.yaml
```

This basic pipeline demonstrates reading from Parquet files and writing back to Parquet format.

_Note: We do not provide any dataset in this repository. You need to acquire the datasets according to their names._

## Docker (Local Mode Only)

This Docker setup is intended for **Spark local mode** only and pins Spark to **3.5.8**. For YARN/cluster usage, continue to use `script/run_yarn.sh` outside Docker.

### Build
```bash
make build
```

### Prepare Small Local Sample (from Hugging Face)
```bash
make prepare-sample
```

### Prepare Assets For All `example/*.yaml`
```bash
make prepare-examples
```

### Run (Default Example)
```bash
make run
```

### Run All Example Configs
```bash
make run-examples
```

### Run (Custom Config)
```bash
PIPELINE_CONFIG=/workspace/<path>.yaml make run
```

Optional sample size override:
```bash
SAMPLE_ROWS=500 make prepare-sample
```

Optional offline prep (skip Hugging Face download):
```bash
SOURCE_MODE=synthetic make prepare-examples
```

## Framework Architecture

### Core Components (`datafiner/`)

The framework provides a comprehensive set of pipeline nodes. Some examples:

| Component           | File                       | Description                                                            |
| ------------------- | -------------------------- | ---------------------------------------------------------------------- |
| **Base Structure**  | `base.py`                  | Abstract classes for custom node implementation                        |
| **Filtering**       | `filter.py`                | Quality-based data filtering with configurable thresholds              |
| **Selection**       | `selector.py`              | Column selection and data projection                                   |
| **Sorting**         | `reorder.py`               | Single and multi-level sorting operations                              |
| **Deduplication**   | `deduplication/minhash.py` | MinHash-based near-duplicate detection                                 |
| **Group Mixing**    | `group_reorder.py`         | Stratified data mixing (see [paper](https://arxiv.org/abs/2512.07612)) |
| **Quality Scoring** | `text_scorer.py`           | FastText-based text quality assessment                                 |
| **I/O Operations**  | `reader.py`, `writer.py`   | Parquet, JSON, and custom format support                               |

Other specific pipeline nodes definition can see `datafiner/`. All nodes inherit from base classes in `base.py`, making it straightforward to implement custom operations.

### Example Configurations (`example/`)

Starter templates for common operations. Some examples:

- **`read_write.yaml`**: Basic I/O pipeline (read from and write to Parquet files)
- **`dedup.yaml`**: Deduplication pipeline using MinHash
- **`filter.yaml`**: Quality filtering based on score metrics
- **`reorder.yaml`**: Data sorting and shuffling examples

Other specific pipeline nodes definition can see `example/`.

### Execution Scripts (`script/`)

Two deployment modes supported:

- **`run_local.sh`**: Local mode for development and testing
- **`run_yarn.sh`**: YARN cluster mode for production workloads

## Production Configuration (`configs/`)

This directory contains the complete preprocessing pipeline used for PCMind-2.1-Kaiyuan-2B training data. The configuration is organized by processing stage:

### 1. Data Cleaning (`clean_filter/`)

Chinese text cleaning pipeline removing:

- Toxic content
- Slang and informal language
- Low-quality advertisements

### 2. Deduplication (`dedup/`)

Near-duplicate removal for major datasets:

- **DCLM-Baseline**: Deduplication of base training corpus
- **Fineweb-Edu-Chinese-V2.1**: Educational content deduplication
- **FinePDF**: Document-level deduplication

### 3. Quantile Selection (`quantile/`)

Score-based data sampling around target quality percentiles

### 4. Tokenization (`tokenization/`)

Tokenization pipelines for various source datasets (JSON, Parquet)

### 5. Phase Construction (`phases/`)

Multi-phase training data preparation:

- **`mix.yaml`**: Data mixing strategies per training phase
- **`count.yaml`**: Token counting and statistics

### 6. Detokenization (`detokenization/`)

Converting tokenized data back to text format for analysis

These configurations provide a **complete recipe** for reproducing the PCMind-2.1-Kaiyuan-2B training pipeline. Use them as reference for building custom preprocessing workflows.

## Usage

You can modify the environment variables in the following scripts to suit your needs.

### YARN Cluster Mode (Production)

```bash
bash script/run_yarn.sh main.py /path/to/config.yaml
```

### Local Mode (Development)

```bash
bash script/run_local.sh main.py /path/to/config.yaml
```

### Configuration File Structure

Example pipeline configuration:

```yaml
spark:
  app_name: my_preprocessing_pipeline

pipeline:
  type: ParquetWriter
  output_path: /output/path
  child_configs:
    - type: Filter
      filter_col: quality_score
      threshold: 0.7
      child_configs:
        - type: ParquetReader
          input_path: /input/path/*.parquet
```

Pipelines are defined as trees where:

- **Leaf nodes**: Data readers (ParquetReader, JsonReader, etc.)
- **Internal nodes**: Transformations (Filter, Dedup, Reorder, etc.)
- **Root node**: Data writers (ParquetWriter, etc.)

## Advanced Features

### Custom Pipeline Nodes

Extend the framework by inheriting from `PipelineNode`:

```python
from datafiner.base import PipelineNode
from datafiner.register import register

@register("CustomNode")
class CustomNode(PipelineNode):
    def __init__(self, spark, custom_param, child_configs=None):
        super().__init__(spark, child_configs)
        self.custom_param = custom_param

    def run(self):
        df = self.children[0].run()
        # Your custom logic here
        return transformed_df
```

## Citation

If you use this framework in your research, please cite:

```bibtex
@article{luo2025pcmind21kaiyuan2btechnicalreport,
  title={PCMind-2.1-Kaiyuan-2B Technical Report},
  author={Kairong Luo and Zhenbo Sun and Xinyu Shi and Shengqi Chen and Bowen Yu anYunyi Chen and Chenyi Dang and Hengtao Tao and Hui Wang and Fangming Liu and KaifenLyu and Wenguang Chen},
  year={2025},
  eprint={2512.07612},
  archivePrefix={arXiv},
  primaryClass={cs.CL},
  url={https://arxiv.org/abs/2512.07612},
}
```

## License

This repository is licensed under [Apache-2.0 License](LICENSE) with the following copyright notice:

```text
Copyright 2025 Tsinghua University & Peng Cheng Laboratory

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
