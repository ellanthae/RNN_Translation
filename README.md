# English-to-Spanish RNN Machine Translation

An English-to-Spanish machine translation system built with PyTorch, PyTorch Lightning, SentencePiece, Weights & Biases, and `uv`.

The model uses:

- a two-layer bidirectional GRU encoder;
- Bahdanau additive attention;
- a two-layer GRU decoder;
- teacher forcing during training;
- greedy autoregressive decoding during inference.

## Dataset

The project uses English-Spanish sentence pairs from the [Tatoeba corpus distributed by ManyThings](https://www.manythings.org/anki/).

- Raw file: `spa.txt`
- License: CC BY 2.0 FR
- Attribution: Tatoeba and ManyThings
- Split seed: `42`

The preprocessing pipeline applies Unicode and whitespace normalization, removes empty and duplicate pairs, filters long sequences and extreme length ratios, and creates deterministic splits.

| Split | Sentence pairs |
|---|---:|
| Train | 115,258 |
| Validation | 14,407 |
| Test | 14,408 |
| Total | 144,073 |

> Tatoeba/ManyThings is not one of the assignment's recommended datasets. Instructor approval may be required.

## Project Structure

```text
.
├── data/                   # Preparation, tokenization, dataset and batching
├── model/                  # Encoder, attention, decoder and Seq2Seq model
├── training/               # Configuration, Lightning training and evaluation
├── artifacts/
│   ├── data/               # Processed splits
│   ├── tokenizers/         # SentencePiece models and vocabularies
│   └── checkpoints/        # Trained Lightning checkpoints
├── main.py                 # Interactive translation CLI
├── pyproject.toml
├── uv.lock
└── README.md
```

## Setup

Requirements:

- Python 3.10+
- `uv`
- Optional CUDA-compatible GPU

Install dependencies:

```powershell
uv sync
```

## Data Preparation

1. Download `spa-eng.zip` from [ManyThings](https://www.manythings.org/anki/).
2. Extract `spa.txt` to:

```text
artifacts/raw_data/spa.txt
```

3. Prepare the data:

```powershell
uv run python -m data.prepare_data
```

4. Train the English and Spanish SentencePiece tokenizers:

```powershell
uv run python -m data.tokenization
```

5. Serialize the vocabulary mappings:

```powershell
uv run python -m data.vocab
```

SentencePiece uses a 16,000-token BPE vocabulary for each language.

| Special token | ID |
|---|---:|
| `<pad>` | 0 |
| `<unk>` | 1 |
| `<bos>` | 2 |
| `<eos>` | 3 |

## Training

Run a one-batch pipeline test:

```powershell
uv run python -m training.train --fast-dev-run
```

Start training with online W&B logging:

```powershell
uv run python -m training.train
```

Train without immediate W&B synchronization:

```powershell
uv run python -m training.train --offline
```

Resume from the latest checkpoint:

```powershell
uv run python -m training.train --resume artifacts/checkpoints/last.ckpt
```

Training includes:

- PAD-masked cross-entropy loss;
- teacher forcing;
- Adam optimizer;
- learning-rate scheduling;
- gradient clipping;
- early stopping;
- best and last checkpoints;
- W&B metric and configuration logging.

Main hyperparameters:

| Parameter | Value |
|---|---:|
| Embedding dimension | 256 |
| Hidden size | 512 |
| GRU layers | 2 |
| Dropout | 0.2 |
| Batch size | 64 |
| Learning rate | 0.001 |
| Teacher-forcing ratio | 0.5 |
| Maximum sequence length | 60 |
| Epochs | 3 |
| Seed | 42 |

The hyperparameters can be changed in `training/config.py`.

On the tested CPU, one epoch required approximately 79 minutes. A CUDA GPU is recommended.

## Evaluation

Evaluate the best checkpoint:

```powershell
uv run python -m training.evaluate --checkpoint artifacts/checkpoints/epoch-02-val_loss-3.2653.ckpt
```

Final results on all 14,408 test sentences:

| Metric | Result |
|---|---:|
| Test loss | 3.2324 |
| Token accuracy | 43.43% |
| BLEU | 28.93 |
| chrF | 51.03 |

Example:

```text
English: Please sit here.
Reference: Sentaos aquí, por favor.
Prediction: Por favor, siéntate aquí.
```

Best checkpoint:

```text
artifacts/checkpoints/epoch-02-val_loss-3.2653.ckpt
```

Best validation loss: `3.2653`.

Checkpoint files may be excluded from Git because of their size. The model can be reproduced using the documented training commands and `uv.lock`.

## Interactive Translator

Run the interactive CLI:

```powershell
uv run python main.py --ckpt artifacts/checkpoints/epoch-02-val_loss-3.2653.ckpt
```

Example session:

```text
Device: cpu
Çıxmaq üçün 'exit' yazın.

Enter English text (or 'exit'): Please sit here.
Spanish: Por favor, siéntate aquí.

Enter English text (or 'exit'): exit
Program bağlandı.
```

The CLI supports:

- configurable checkpoint paths;
- CUDA when available;
- CPU fallback;
- empty-input handling;
- long-input truncation;
- `<eos>` stopping;
- `exit` command.

## Weights & Biases

Project:

https://wandb.ai/belnaz456-khazar-university/rnn-en-es-translation

Training run:

https://wandb.ai/belnaz456-khazar-university/rnn-en-es-translation/runs/ghtzs9ay

Evaluation run:

https://wandb.ai/belnaz456-khazar-university/rnn-en-es-translation/runs/06gqszlg

The runs contain losses, token accuracy, learning rate, configuration, BLEU, chrF, and sample translations.

## Reproducibility

The project uses:

- deterministic data splitting;
- fixed seed `42`;
- deterministic Lightning training;
- serialized tokenizers and vocabularies;
- saved checkpoints;
- locked dependencies in `uv.lock`;
- logged W&B configuration.

## Known Limitations

- Greedy decoding can produce suboptimal translations.
- The model sometimes repeats words or produces incomplete sentences.
- Input capitalization affects tokenization and translation.
- The Tatoeba dataset contains some noisy references.
- Three epochs are insufficient for fully stable translation.
- Beam search and teacher-forcing decay are not implemented.
- Training is slow without a GPU.

## Quality Checks

```powershell
uv run ruff format --check .
uv run ruff check .
uv run python -c "import data, model, training; print('Imports passed')"
```

## Run Everything From Scratch

```powershell
uv sync
uv run python -m data.prepare_data
uv run python -m data.tokenization
uv run python -m data.vocab
uv run python -m training.train
uv run python -m training.evaluate --checkpoint artifacts/checkpoints/last.ckpt
uv run python main.py --ckpt artifacts/checkpoints/last.ckpt
```