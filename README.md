# sparv-bert-neighbour-plugin

[![CI](https://github.com/spraakbanken/sparv-bert-neighbour-plugin/actions/workflows/ci.yml/badge.svg)](https://github.com/spraakbanken/sparv-bert-neighbour-plugin/actions/workflows/ci.yml)
[![PyPI version](https://badge.fury.io/py/sparv-bert-neighbour-plugin.svg)](https://pypi.org/project/sparv-bert-neighbour-plugin)

Plugin for applying bert masking as a [Sparv](https://github.com/spraakbanken/sparv-pipeline) annotation.

## Install

First, install Sparv, as suggested:

```bash
pipx install sparv-pipeline
```

Then install install `sparv-bert-neighbour-plugin` with

```bash
pipx inject sparv-pipeline sparv-bert-neighbour-plugin
```

## Usage
Depending on how many exlicit exports of annotations you have you can decide to use this 
annotation exclusively by adding it as the only annotation to export under `xml_export`:

```yaml
xml_export:
    annotations:
        - <token>:sparv_bert_neighbour.transformer-neighbour
```
To use it together with other annotations you might add it under `export`:

```yaml
export:
    annotations:
        - <token>:sparv_bert_neighbour.transformer-neighbour
        ...
```

### Configuration

You can configure this plugin by choosing a huggingface model, huggingface transformer and the number of neighbours to generate.

#### Model

The model defaults to [`KBLab/bert-base-swedish-cased`](https://huggingface.co/KBLab/bert-base-swedish-cased) but can be configured in `config.yaml`:

```yaml
sparv_bert_neighbour:
    model: "KBLab/bert-base-swedish-cased"
```

#### Tokenizer

The tokenizer defaults to [`KBLab/bert-base-swedish-cased`](https://huggingface.co/KBLab/bert-base-swedish-cased) but can be configured in `config.yaml`:

```yaml
sparv_bert_neighbour:
    tokenizer: "KBLab/bert-base-swedish-cased"
```

#### Number of Neighbours

The number of neighbours defaults to `5` but can be configured in `config.yaml`:

```yaml
sparv_bert_neighbour:
    num_neighbours: 5
```
