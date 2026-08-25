# English-to-Spanish RNN Machine Translation

English-to-Spanish neural machine translation using PyTorch, PyTorch Lightning, SentencePiece and Weights & Biases.

## Architecture

- Two-layer bidirectional GRU encoder
- Bahdanau additive attention
- Two-layer GRU decoder
- Teacher forcing during training
- Beam Search with length normalization during inference
- Consecutive-token repetition blocking

## Dataset

English-Spanish sentence pairs are taken from the [Tatoeba/ManyThings dataset](https://www.manythings.org/anki/).

Tatoeba/ManyThings was selected because it is manageable in size,
contains conversational sentence pairs, and is suitable for training
an RNN baseline with limited computational resources.

| Split | Sentence pairs |
|---|---:|
| Train | 115,258 |
| Validation | 14,407 |
| Test | 14,408 |
| Total | 144,073 |

Preprocessing removes empty and duplicate pairs, normalizes text, filters long sequences and creates deterministic splits using seed `42`.

## Project Structure

```text
data/           Data preparation, tokenization, dataset and batching
model/          Encoder, Bahdanau attention, decoder and Seq2Seq
training/       Configuration, Lightning training and evaluation
main.py         Interactive translation
artifacts/      Data, tokenizers and checkpoints (excluded from Git)
```

## Setup

Requirements:

- Python 3.14+
- `uv`
- Optional CUDA-compatible GPU

```bash
uv sync
```

Download `spa-eng.zip` from ManyThings and place the extracted file at:

```text
artifacts/raw_data/spa.txt
```

Prepare the data and train the SentencePiece tokenizers:

```bash
uv run python -m data.prepare_data
uv run python -m data.tokenization
uv run python -m data.vocab
```

Separate 16,000-token BPE vocabularies are used for English and Spanish.

| Token | ID |
|---|---:|
| `<pad>` | 0 |
| `<unk>` | 1 |
| `<bos>` | 2 |
| `<eos>` | 3 |

## Training

Main hyperparameters:

| Parameter | Value |
|---|---:|
| Embedding dimension | 256 |
| Hidden size | 512 |
| GRU layers | 2 |
| Dropout | 0.3 |
| Batch size | 64 |
| Initial learning rate | 0.0005 |
| Teacher-forcing ratio | 0.5 |
| Maximum length | 60 |
| Vocabulary size | 16,000 |
| Gradient clipping | 1.0 |
| Early-stopping patience | 5 |

Run a one-batch test:

```bash
uv run python -m training.train --fast-dev-run --offline
```

Start training:

```bash
uv run python -m training.train --offline
```

Resume interrupted training:

```bash
uv run python -m training.train \
  --offline \
  --resume artifacts/checkpoints/last.ckpt
```

The best model was selected at epoch 5:

```text
artifacts/checkpoints_exp2/epoch-05-val_loss-2.9262.ckpt
```

Best validation loss: `2.9262`.

## Evaluation

```bash
uv run python -m training.evaluate \
  --checkpoint artifacts/checkpoints/epoch-05-val_loss-2.9262.ckpt \
  --beam-size 5 \
  --length-penalty 0.6 \
  --offline
```

Results on all 14,408 test sentences:

| Metric | Result |
|---|---:|
| Test loss | 2.8963 |
| Token accuracy | 47.31% |
| BLEU | 38.88 |
| chrF | 58.97 |

## Interactive Translation

```bash
uv run python main.py \
  --ckpt artifacts/checkpoints/epoch-05-val_loss-2.9262.ckpt \
  --beam-size 5 \
  --length-penalty 0.6
```

Example:

```text
English: Where is the train station?
Spanish: ¿Dónde está la estación de trenes?

English: Although he was tired, he continued working until midnight.
Spanish: Aunque estaba cansado, él siguió trabajando hasta la medianoche.
```

## Experiment Tracking

Training configuration and metrics are logged in the [W&B project](https://wandb.ai/belnaz456-khazar-university/rnn-en-es-translation).

Final evaluation run:
https://wandb.ai/belnaz456-khazar-university/rnn-en-es-translation/runs/ad1bd6tf

Logged information includes loss, token accuracy, learning rate, BLEU, chrF and sample translations.

## Reproducibility

The project uses:

- fixed seed `42`;
- deterministic dataset splitting;
- serialized SentencePiece tokenizers;
- Lightning checkpoints;
- locked dependencies in `uv.lock`;
- W&B configuration logging.

The `artifacts/` directory is excluded from Git because it contains the dataset, tokenizers and large checkpoints. The complete pipeline can be reproduced using the documented commands.

## Limitations

- Long or complex sentences may lose information
- Rare words may produce incorrect subword combinations
- The dataset contains some noisy or uncommon translations
- Beam Search is slower than greedy decoding

## Quality Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run python -c "import data, model, training; print('Imports passed')"
```