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

6167652d656e6372797074696f6e2e6f72672f76310a2d3e20736372797074207a656d436e4c4c78765047666e43362f587044376b412031380a7161396d38716b42516259707833424a4947734158754c63302f456d615150485675356856357a73354f490a2d2d2d204c625a6a4d4c613873664e676b777569776b614738556a663348363934704c593165666f484332763859550aa3249e5de404da772ecc434222b10b2b8d8a92dc31a15df728502bc9995b8d5d0ac223efe57cbe708f19cadaa04bf8c67dbe5afa9575ccd5571ba6a2b10225b608d02b03204f3555c98182f2afb78654a7fc123ee31d574ae9f838f16d8638ce4a3e758e542fb23909cf22ba99991362132a0d7c81e34f60f0f16e581f01ed171b3a3efd23d1a6f3ac9c722da324ee2285db5a85a98372a022e3e375f194b7aab5412f50d940cbfd2bb303f2abfd2527ec0e4533f468f5b5a8acbd4289d11af9e0c7bdad2f1740d08ba97369592cc6274edecf

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
