# English-to-Spanish RNN Machine Translation

An English-to-Spanish neural machine translation system implemented with PyTorch, PyTorch Lightning, SentencePiece, Weights & Biases, and `uv`.

The model uses:

- a two-layer bidirectional GRU encoder;
- Bahdanau additive attention;
- a two-layer GRU decoder;
- teacher forcing during training;
- Beam Search during inference;
- length normalization and consecutive-token repetition blocking.

## Dataset

The project uses English-Spanish sentence pairs from the [Tatoeba corpus distributed by ManyThings](https://www.manythings.org/anki/).

Preprocessing includes:

- Unicode and whitespace normalization;
- removal of empty and duplicate sentence pairs;
- maximum-length filtering;
- extreme length-ratio filtering;
- deterministic train, validation, and test splitting.

| Split | Sentence pairs |
|---|---:|
| Train | 115,258 |
| Validation | 14,407 |
| Test | 14,408 |
| Total | 144,073 |

Split seed: `42`.

> The dataset is distributed by ManyThings and originates from Tatoeba. Its listed license is CC BY 2.0 FR.

## Project Structure

```text
.
├── data/
│   ├── prepare_data.py
│   ├── tokenization.py
│   ├── vocab.py
│   ├── dataset.py
│   └── collate.py
├── model/
│   ├── encoder.py
│   ├── attention.py
│   ├── decoder.py
│   └── seq2seq.py
├── training/
│   ├── config.py
│   ├── lightning_module.py
│   ├── train.py
│   └── evaluate.py
├── artifacts/
│   ├── raw_data/
│   ├── data/
│   ├── tokenizers/
│   └── checkpoints/
├── main.py
├── pyproject.toml
├── uv.lock
└── README.md
```

The `artifacts/` directory is excluded from Git because it contains datasets, tokenizers, and large checkpoint files.

## Setup

Requirements:

- Python 3.14+
- `uv`
- optional CUDA-compatible GPU

Install the locked dependencies:

```bash
uv sync
```

If the server requires system TLS certificates:

```bash
uv sync --system-certs
```

## Data Preparation

Download `spa-eng.zip` from [ManyThings](https://www.manythings.org/anki/) and extract `spa.txt` to:

```text
artifacts/raw_data/spa.txt
```

Prepare deterministic dataset splits:

```bash
uv run python -m data.prepare_data
```

Train the English and Spanish SentencePiece tokenizers:

```bash
uv run python -m data.tokenization
```

Create vocabulary information:

```bash
uv run python -m data.vocab
```

Separate 16,000-token BPE vocabularies are used for English and Spanish.

| Special token | ID |
|---|---:|
| `<pad>` | 0 |
| `<unk>` | 1 |
| `<bos>` | 2 |
| `<eos>` | 3 |

## Model Architecture

The source sentence is encoded using a two-layer bidirectional GRU. The decoder is a two-layer unidirectional GRU that uses Bahdanau attention over all encoder outputs.

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
| Vocabulary size | 16,000 per language |
| Gradient clipping | 1.0 |
| Early-stopping patience | 5 |
| Seed | 42 |

The model contains approximately 30.1 million trainable parameters.

## Training

Run a one-batch pipeline test:

```bash
uv run python -m training.train --fast-dev-run
```

Train with online W&B logging:

```bash
uv run python -m training.train
```

Train with offline W&B logging:

```bash
uv run python -m training.train --offline
```

Resume a stopped training run:

```bash
uv run python -m training.train \
  --resume artifacts/checkpoints/last.ckpt
```

Training uses:

- PAD-masked cross-entropy loss;
- teacher forcing;
- Adam optimization;
- learning-rate scheduling;
- gradient clipping;
- validation-based early stopping;
- best and latest checkpoint saving;
- W&B experiment tracking.

Training stopped at epoch 12 because validation loss had not improved for five consecutive validation records. The best checkpoint was obtained at epoch 7:

```text
artifacts/checkpoints/epoch-07-val_loss-3.1066.ckpt
```

Best validation loss:

```text
3.1066
```

## Evaluation

Evaluate using Beam Search:

```bash
uv run python -m training.evaluate \
  --checkpoint artifacts/checkpoints/epoch-07-val_loss-3.1066.ckpt \
  --beam-size 5 \
  --length-penalty 0.6 \
  --offline
```

Final results on all 14,408 test sentences:

| Metric | Result |
|---|---:|
| Test loss | 3.0726 |
| Token accuracy | 46.26% |
| BLEU | 37.88 |
| chrF | 58.31 |

The final decoding configuration uses:

- Beam Search size: `5`
- length penalty: `0.6`
- consecutive-token repetition blocking
- maximum generated length: `60`

Comparison with greedy decoding:

| Decoding method | BLEU | chrF |
|---|---:|---:|
| Greedy decoding | 34.79 | 55.89 |
| Beam Search | **37.88** | **58.31** |

Beam Search improved BLEU by approximately `3.09` points and chrF by approximately `2.43` points.

## Interactive Translation

Run the interactive translator:

```bash
uv run python main.py \
  --ckpt artifacts/checkpoints/epoch-07-val_loss-3.1066.ckpt \
  --beam-size 5 \
  --length-penalty 0.6
```

Example:

```text
Enter English text (or 'exit'): I am a student.
Spanish: Soy un estudiante.

Enter English text (or 'exit'): The weather is good today.
Spanish: Hoy hace buen tiempo.

Enter English text (or 'exit'): I want to learn Spanish.
Spanish: Quiero aprender español.

Enter English text (or 'exit'): We went to the store yesterday.
Spanish: Ayer fuimos a la tienda.
```

The CLI supports:

- configurable checkpoint paths;
- configurable Beam Search parameters;
- CUDA when available;
- CPU fallback;
- empty-input validation;
- long-input truncation;
- EOS-based stopping;
- the `exit` command.

## Weights & Biases

W&B project:

https://wandb.ai/belnaz456-khazar-university/rnn-en-es-translation

The project records:

- training and validation loss;
- token accuracy;
- learning rate;
- model configuration;
- BLEU and chrF;
- sample translations.

An offline run can be synchronized later using:

```bash
uv run wandb sync wandb/offline-run-<run-id>
```

## Reproducibility

Reproducibility is supported through:

- fixed random seed `42`;
- deterministic dataset splitting;
- saved SentencePiece models;
- PyTorch Lightning checkpoints;
- configuration logging;
- W&B experiment tracking;
- dependency locking through `uv.lock`.

To reproduce the complete pipeline:

```bash
uv sync
uv run python -m data.prepare_data
uv run python -m data.tokenization
uv run python -m data.vocab
uv run python -m training.train
```

After training, use the best checkpoint path printed by the training command:

```bash
uv run python -m training.evaluate \
  --checkpoint <best-checkpoint-path> \
  --beam-size 5 \
  --length-penalty 0.6
```

## Known Limitations

- Long and syntactically complex sentences may lose information.
- Rare words may be translated incorrectly or produce malformed subwords.
- The Tatoeba/ManyThings corpus contains noisy or uncommon reference translations.
- Beam Search improves sequence selection but cannot correct knowledge the model did not learn.
- Beam Search is slower than greedy decoding.
- Translation quality is sensitive to the coverage and quality of the training dataset.

## Quality Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run python -c "import data, model, training; print('Imports passed')"
```