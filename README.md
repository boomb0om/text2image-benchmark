![](assets/logo.png)

This project aims to unify the evaluation of generative text-to-image models and provide the ability to quickly and easily calculate most popular metrics.

Goals of this benchmark:
- **Unified** metrics and datasets for all models
- **Reproducible** results
- **User-friendly** interface for most popular metrics: FID and CLIP-score

## Table of Contents

- [Introduction](#introduction)
- [Main features](#main-features)
- [Installation](#installation)
- [Getting started](#getting-started)
- [Project Structure](#project-structure)
- [Examples](#examples)
- [Documentation](#documentation)
- [Contribution](#contribution)
- [Contacts](#contacts)
- [Citing](#citing)
- [Acknowledgments](#acknowledgments)

## Introduction

Generative text-to-image models have become a popular and widely used tool for users. 
There are many articles on the topic of image generation from text that present new, more advanced models.
However, there is still no uniform way to measure the quality of such models.
To address this issue, we provide an implementation of metrics to compare the quality of generative models.

We propose to use the metric MS-COCO FID-30K with OpenAI's CLIP score, which has already become a standard for measuring the quality of text2image models. 
We provide the MS-COCO validation subset and precalculated metrics for it. 
We also recorded 30,000 descriptions that needs to be used to generate images for MS-COCO FID-30K.

You can easily contribute your model into benchmark and make FID results reproducible! See more in [contribution](#contribution) section.

## Main features

- Standardized FID calculation: fixed image preprocessing and InceptionV3 model.
- FID-30k on MS-COCO validation set: we provide dataset on [huggingface🤗](https://huggingface.co/datasets/stasstaf/MS-COCO-validation), [precomputed FID stats](https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_fid_stats.npz), fixed [30000 captions from MS-COCO](https://github.com/boomb0om/text2image-benchmark/releases/download/v0.0.1/MS-COCO_val2014_30k_captions.csv) that should be used to generate images
- CLIP-score calculation
- User-friendly metrics calculation (checkout [Getting started](#getting-started))

## Installation

```bash
pip install git+https://github.com/boomb0om/text2image-benchmark
```

## Getting started


### Metrics: FID

Calculate FID for two sets of images:

```python
from T2IBenchmark import calculate_fid

fid, _ = calculate_fid('assets/images/cats/', 'assets/images/dogs/')
print(fid)
```

Calculate FID between model generations and MS-COCO validation subset:

```python
from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats

fid, _ = calculate_fid(
    'path/to/your/generations/',
    get_coco_fid_stats()
)
```

MS-COCO FID-30k for T2IModelWrapper. In this example we are using [Kandinsky 2.1](https://github.com/ai-forever/Kandinsky-2) model:

```bash
pip install -r T2IBenchmark/models/kandinsky21/requirements.txt
```

```python
from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats

fid, _ = calculate_fid(
    'path/to/your/generations/',
    get_coco_fid_stats()
)
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
- `assets/` - Assets

## Examples



## Documentation



## Contribution



## Contacts

If you have any question, please email `jeartgle@gmail.com`.

## Citing

If you use this repository in your research, consider citing it using the following Bibtex entry:

```
@misc{boomb0omT2IBenchmark,
  author={Pavlov, I. and Ivanov, A. and Stafievskiy, S.},
  title={{Text-to-Image Benchmark: A benchmark for generative models}},
  howpublished={\url{https://github.com/boomb0om/text2image-benchmark}},
  month={September},
  year={2023},
  note={Version 0.1.0},
}
```

## Acknowledgments

Thanks to:

- [clean-fid](https://github.com/GaParmar/clean-fid/) - Explanation of influence of various parameters when calculating FID.
- [pytorch-fid](https://github.com/mseitzer/pytorch-fid) - Port of the official implementation of Frechet Inception Distance to PyTorch.