# blueport-cv

> Computer-vision pipeline for automated detection and classification of plastic and waste in port and riverine environments, using CLIP.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org)

## Overview

`blueport-cv` is the open code behind [BluePort AI](https://blueportai.manus.space), a tool for automated environmental monitoring through image classification. It originated in the Omdena AI for the Ganges River project and was extended into a production-ready computer-vision workflow for port waste monitoring.

## Features

- Zero-shot and fine-tuned classification with CLIP
- Image preprocessing and batching utilities
- Training and evaluation notebooks
- Sample annotated images for quick testing
- Exportable predictions for downstream dashboards

## Quick start

```python
from blueport_cv import classify_image

label, score = classify_image("samples/debris_01.jpg")
print(label, round(score, 3))
```

## Repository structure

```
blueport-cv/
├── blueport_cv/         # inference + training code
├── notebooks/           # training and evaluation
├── data/sample/         # sample images (annotated)
└── models/              # model cards and checkpoints (or links)
```

## Acknowledgements

Built on work from the Omdena AI for the Ganges River initiative (10+ countries, open-science framework).

## License

MIT
