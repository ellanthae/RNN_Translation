# English-to-Spanish RNN Machine Translation

English-to-Spanish neural machine translation using PyTorch, PyTorch Lightning, SentencePiece, and Weights & Biases.

## Architecture

- Two-layer bidirectional GRU encoder
- Bahdanau additive attention
- Two-layer GRU decoder
- Teacher forcing during training
- Beam Search with length normalization during inference
- Consecutive-token repetition blocking

## Dataset

The final model was trained on English–Spanish sentence pairs from the [Tatoeba/ManyThings dataset](https://www.manythings.org/anki/).

Tatoeba was selected because it contains conversational sentence pairs, is manageable with limited computational resources, and is well suited to a Seq2Seq RNN baseline.

| Split | Sentence pairs |
|---|---:|
| Train | 115,258 |
| Validation | 14,407 |
| Test | 14,408 |
| Total | 144,073 |

Preprocessing removes empty and duplicate pairs, normalizes text, filters long sequences, and creates deterministic splits using seed `42`.

## Project Structure

```text
data/           Data preparation, tokenization, dataset, and batching
model/          Encoder, Bahdanau attention, decoder, and Seq2Seq
training/       Configuration, Lightning training, and evaluation
main.py         Interactive translation
artifacts/      Data, tokenizers, and checkpoints (excluded from Git)
```

## Setup

Requirements:

- Python 3.14+
- `uv`
- Optional CUDA-compatible GPU

Install the locked dependencies:

```bash
uv sync
```

Download `spa-eng.zip` from ManyThings and place the extracted file at:

```text
artifacts/raw_data/spa.txt
```

Prepare the data and build the SentencePiece artifacts:

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

## Training Configuration

| Parameter | Value |
|---|---:|
| Embedding dimension | 256 |
| Hidden size | 512 |
| GRU layers | 2 |
| Bidirectional encoder | Yes |
| Dropout | 0.35 |
| Batch size | 64 |
| Learning rate | 0.0005 |
| Teacher-forcing ratio | 0.5 |
| Maximum sequence length | 60 |
| Vocabulary size | 16,000 per language |
| Gradient clipping | 1.0 |
| Early-stopping patience | 5 |
| Random seed | 42 |

Run a one-batch pipeline test:

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

## Final Model

The final checkpoint was selected using validation loss and translation-quality metrics:

```text
epoch-09-val_loss-2.9307.ckpt
```

Final Tatoeba results:

| Metric | Result |
|---|---:|
| Best epoch | 9 |
| Validation loss | 2.9307 |
| Test loss | 2.8922 |
| Token accuracy | 48.10% |
| BLEU | 40.18 |
| chrF | 60.44 |
| Evaluated sentences | 14,408 |

Inference uses Beam Search with:

```text
beam_size = 5
length_penalty = 0.6
```

## Evaluation

```bash
uv run python -m training.evaluate \
  --checkpoint artifacts/checkpoints/epoch-09-val_loss-2.9307.ckpt \
  --beam-size 5 \
  --length-penalty 0.6 \
  --offline
```

## Dataset Experiments

Additional dataset experiments were conducted to study the effect of corpus size, domain, and data splitting.

| Dataset | Validation loss | Test loss | Token accuracy | BLEU | chrF |
|---|---:|---:|---:|---:|---:|
| Tatoeba final model | 2.9307 | 2.8922 | 48.10% | 40.18 | 60.44 |
| Kaggle English–Spanish | 2.9414 | 3.1023 | 46.22% | 39.26 | 58.62 |
| OPUS-100 150K | 5.6198 | 5.6623 | 17.04% | 14.63 | 36.31 |

The Kaggle dataset was split by unique normalized English source sentences. The resulting train, validation, and test source overlaps were all zero.

The OPUS-100 experiment used a deterministic 150,000-pair training subset and the official validation and test sets. Its mixed legal, diplomatic, subtitle, and conversational domains were substantially more difficult for the GRU baseline.

Metrics across different test corpora are not perfectly comparable. However, the experiments show that domain suitability and data quality are more important than dataset size alone. The Tatoeba model was selected as the final model because it achieved the strongest overall translation metrics.

## Interactive Translation

```bash
uv run python main.py \
  --ckpt artifacts/checkpoints/epoch-09-val_loss-2.9307.ckpt \
  --beam-size 5 \
  --length-penalty 0.6
```

Examples:

```text
English: I am a student.
Spanish: Soy un estudiante.

English: Where is the train station?
Spanish: ¿Dónde está la estación de trenes?

English: Although he was tired, he continued working until midnight.
Spanish: Aunque estaba cansado, él siguió trabajando hasta la medianoche.
```

## Experiment Tracking

Training configurations and metrics are tracked in the [Weights & Biases project](https://wandb.ai/belnaz456-khazar-university/rnn-en-es-translation).

Logged information includes:

- Training and validation loss
- Token accuracy
- Learning rate
- BLEU
- chrF
- Configuration values
- Sample translations

## Model Checkpoint

Model checkpoints are excluded from Git because they exceed GitHub's standard file-size limit. The final checkpoint must be stored externally together with its matching SentencePiece tokenizer files.

Required artifacts:

```text
final_model/
├── epoch-09-val_loss-2.9307.ckpt
└── tokenizers/
    ├── source.model
    ├── source.vocab
    ├── target.model
    └── target.vocab
```

## Reproducibility

The project uses:

- Fixed random seed `42`
- Deterministic dataset splitting
- Serialized SentencePiece tokenizers
- PyTorch Lightning checkpoints
- Locked dependencies in `uv.lock`
- W&B configuration logging
- Source-disjoint splits for the Kaggle experiment

The `artifacts/` directory is excluded from Git because it contains datasets, tokenizers, and large checkpoints.

## Limitations

- Long or complex sentences may lose information.
- Rare words may produce incorrect subword combinations.
- The training corpus contains noisy or uncommon translations.
- Multiple valid translations may be penalized by single-reference BLEU.
- Beam Search is slower than greedy decoding.
- The GRU baseline is less scalable than modern Transformer models.

## Quality Checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run python -c "import data, model, training; print('Imports passed')"
```