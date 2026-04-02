# BluePort AI - Waste Classification Bot

A Telegram bot that classifies waste materials from photos using **CLIP vision** (OpenAI ViT-B/32) with a custom linear probe, running entirely **offline** with no external API calls.

Built as a research prototype exploring AI-driven waste sorting for the Brazilian recycling context.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-red)
![CLIP](https://img.shields.io/badge/Model-CLIP_ViT--B%2F32-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## How It Works

The user sends a photo via Telegram. The bot downloads the image, extracts visual features using CLIP's frozen encoder, and passes them through a lightweight linear probe (16 KB) trained on ~11,400 waste images across 6 categories. The prediction and confidence score are returned in seconds.

```
User sends photo → CLIP encodes image → Linear probe classifies → Bot replies with label + confidence
```

If no trained model is available, the system falls back gracefully to CLIP zero-shot classification using optimised text prompts for 9 waste categories.

## Results

Evaluated on 11,454 images across 6 waste categories:

| Category | Accuracy | Images |
|---|---|---|
| Electronic | 98.1% | 2,543 |
| Paper | 96.4% | 2,237 |
| Metal | 95.6% | 2,247 |
| Glass | 94.3% | 2,036 |
| Plastic | 91.8% | 2,219 |
| Organic | 87.8% | 172 |
| **Overall** | **95.2%** | **11,454** |

The organic category has lower accuracy due to class imbalance (172 vs ~2,200 images for other classes).

## Key Features

**Privacy-first architecture**: all inference runs locally. No images are sent to external services.

**Lightweight domain adaptation**: instead of fine-tuning the full CLIP model, only a 16 KB linear layer is trained on top of frozen CLIP features. This keeps computational costs minimal while achieving 95%+ accuracy.

**Confidence calibration**: softmax temperature scaling and a configurable rejection threshold filter out uncertain predictions.

**Graceful fallback**: if the trained model is unavailable, the system automatically switches to CLIP zero-shot classification with hand-crafted prompts.

**Bilingual support**: categories and interface available in Portuguese (PT-BR) and English, configurable via environment variable.

## Project Structure

```
├── waste_bot.py              # Telegram bot (handlers, commands)
├── waste_vision.py           # CLIP inference engine + logging
├── train_linear_probe.py     # Training script for domain adaptation
├── eval_batch.py             # Batch evaluation with ground truth
├── check_dataset.py          # Dataset validation and cleaning
├── labels.json               # Category taxonomy (PT/EN)
├── blueport_linear.pt        # Trained linear probe weights (16 KB)
├── requirements.txt          # Dependencies
├── .env.example              # Environment config template
└── LICENSE
```

## Setup

**Requirements**: Python 3.10+, pip

```bash
# Clone the repository
git clone https://github.com/darlianecunha/blueport-ai-wastebot.git
cd blueport-ai-wastebot

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env and add your Telegram bot token (from @BotFather)

# Run
python waste_bot.py
```

## Tech Stack

**Vision model**: OpenAI CLIP (ViT-B/32), frozen as feature extractor

**Domain adaptation**: custom linear probe trained on waste images (PyTorch)

**Interface**: Telegram Bot API (python-telegram-bot)

**Data pipeline**: Pillow for image processing, Pandas for logging and analysis, SQLite for persistence

## Dataset

Training and evaluation used ~11,400 images across 6 waste categories (plastic, paper, metal, glass, organic, electronic), sourced from public datasets including TACO and TrashNet. The dataset is not included in this repository due to size (1.9 GB). The `check_dataset.py` utility validates image integrity and quarantines corrupted files.

## Commands

| Command | Description |
|---|---|
| `/start` | Welcome message and instructions |
| `/stats` | Total images analysed + average confidence |
| `/count` | Total image counter |
| Send a photo | Returns waste classification + confidence |

## Possible Extensions

- Web dashboard for monitoring classification metrics
- Integration with IoT smart bins for automated sorting
- Feedback loop for continuous model improvement
- Gravimetric composition reports for PGRS compliance

## Author

**Darliane Cunha** - PhD in Finance and Sustainability

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
