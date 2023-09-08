![](assets/logo.png)

This project aims to unify the evaluation of generative text-to-image models and provide the ability to quickly and easily calculate most popular metrics.

Core features:
- **Reproducible** results
- **Unified** metrics and datasets for all models
- **User-friendly** interface for most popular metrics: FID, IS, CLIP-score

## Table of Contents

- [Introduction](#introduction)
- [Installation](#installation)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Documentation](#documentation)
- [Getting started](#getting-started)
- [License](#license)
- [Acknowledgments](#acknowledgments)
- [Contacts](#contacts)
- [Citation](#citation)
- [Examples of good README's](#examples-of-good-readmes)

## Introduction

Generative text-to-image models have become a popular and widely used tool for users. 
There are many articles on the topic of image generation from text that present new, more advanced models.
However, there is still no uniform way to measure the quality of such models.
To address this issue, we provide an implementation of metrics to compare the quality of generative models.

We propose to use the metric MS-COCO FID-30K with OpenAI's CLIP score, which has already become a standard for measuring the quality of text2image models. 
We provide the MS-COCO validation subset and precalculated metrics for it. 
We also recorded 30,000 descriptions that need to be used to generate images for MS-COCO FID-30K.

## Installation

```bash
pip install git+https://github.com/boomb0om/text2image-benchmark
```

## Getting started

**Calculate FID for two sets of images**:

```python
from T2IBenchmark import calculate_fid

fid, _ = calculate_fid('assets/images/cats/', 'assets/images/dogs/')
print(fid)
```

## Project Structure

- `T2IBenchmark/`
  - `datasets/` - Datasets that can be used for evaluation
    - `coco2014/` - MS-COCO 2014 validation subset
  - `feature_extractors/` - Implementation of different neural nets used to extract features from images
  - `metrics/` - Implementation of metrics
  - `utils/` - Some utils
- `docs/` - Documentation
- `examples/` - Usage examples
- `experiments/` - Experiments

## Documentation

Link to the documentation

## Getting started

Tutorials if any

## License

Link to the license

## Acknowledgments

Acknowledgments

## Contacts

Your contacts. For example:

- [Telegram channel](https://t.me/) answering questions about your project
- [VK group](<https://vk.com/>) your VK group
- etc.

## Citation

@article{"name",
  title = {},
  author = {},
  journal = {},
  year = {},
  issn = {},
  doi = {}}

bibtex-ссылку удобно брать с google scholar

# Examples of good README's

- <https://github.com/pytorch/pytorch>
- <https://github.com/scikit-learn/scikit-learn>
- <https://github.com/aimclub/FEDOT>
