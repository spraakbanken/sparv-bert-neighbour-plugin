# sparv-word-prediction--kb-bert-plugin

[![PyPI version](https://badge.fury.io/py/sparv-word-prediction-kb-bert-plugin.svg)](https://pypi.org/project/sparv-word-prediction-kb-bert-plugin)

Plugin for applying bert masking as a [Sparv](https://github.com/spraakbanken/sparv-pipeline) annotation.

## Install

First, install Sparv, as suggested:

```bash
pipx install sparv-pipeline
```

Then install install `sparv-word-prediction-kb-bert-plugin` with

```bash
pipx inject sparv-pipeline sparv-word-prediction-kb-bert-plugin
```

## Usage

Depending on how many explicit exports of annotations you have you can decide to use this
annotation exclusively by adding it as the only annotation to export under `xml_export`:

```yaml
xml_export:
    annotations:
        - <token>:word_prediction_kb_bert.word-prediction--kb-bert
```

To use it together with other annotations you might add it under `export`:

```yaml
export:
    annotations:
        - <token>:word_prediction_kb_bert.word-prediction--kb-bert
        ...
```

### Configuration

You can configure this plugin by the number of neighbours to generate.

#### Number of Neighbours

The number of neighbours defaults to `5` but can be configured in `config.yaml`:

```yaml
word_prediction_kb_bert:
    num_neighbours: 5
```

### Metadata

#### Model

Type | HuggingFace Model | Revision
--- | --- | ---
Model | [`KBLab/bert-base-swedish-cased`](https://huggingface.co/KBLab/bert-base-swedish-cased) | c710fb8dff81abb11d704cd46a8a1e010b2b022c
Tokenizer | same as Model  | same as Model

## Changelog

This project keeps a [changelog](./CHANGELOG.md).
