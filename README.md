# Can LLMs Design Good Questions Based on Context?

This repository accompanies the paper [*Can LLMs Design Good Questions Based on Context?*]() available on arXiv. It provides datasets, experimental scripts, and visualization tools to explore the capability of Large Language Models (LLMs) in designing meaningful and contextually relevant questions.

## Introduction

This is the code repository for the paper *Can LLMs Design Good Questions Based on Context?* The repository contains experimental scripts, and visualization tools to analyze the performance of LLMs in generating questions based on context.

## Installation

This project uses [Poetry](https://python-poetry.org/) for dependency management.**Install Poetry (if not already installed):**

1. Follow the [official installation guide](https://python-poetry.org/docs/#installation).
2. **Install dependencies:**
  ```bash
  poetry install
  ```
3. Install punkit 
  ```bash
  python setup.py
  ```

## Usage

### Running Experiments

The entire pipeline for running experiments is automated through the `run.py` script. To run the experiments, execute the following command:

```bash
python run.py <experiment> [--flags]
```

### Generating Plots

Visualization scripts are available to generate plots based on experimental results.

```bash
python plot.py <plot_name> [--flags]
```

## Data Description

The `data` directory contains various datasets used for training and evaluating the LLMs:

* **hotpot:** Datasets related to the HotpotQA project.
* **llmqg_gpt & llmqg_llama:** Our QA dataset generated using WikiText and different LLMs.
* **trivia:** Trivia QA datasets, both filtered and unfiltered.
