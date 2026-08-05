import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from data.collate import collate_translation_batch
from data.dataset import TranslationDataset
from data.tokenization import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    SentencePieceTokenizer,
)
from model.decoder import Decoder
from model.encoder import Encoder
from model.seq2seq import Seq2Seq
from training.config import TrainingConfig
from training.lightning_module import TranslationLightningModule


def serialize_config(config: TrainingConfig) -> dict[str, Any]:
    serialized: dict[str, Any] = {}

    for key, value in asdict(config).items():
        serialized[key] = str(value) if isinstance(value, Path) else value

    return serialized


def build_tokenizers(
    config: TrainingConfig,
) -> tuple[SentencePieceTokenizer, SentencePieceTokenizer]:
    source_tokenizer = SentencePieceTokenizer(config.tokenizer_dir / "source.model")
    target_tokenizer = SentencePieceTokenizer(config.tokenizer_dir / "target.model")

    return source_tokenizer, target_tokenizer


def build_dataset(
    config: TrainingConfig,
    split: str,
    source_tokenizer: SentencePieceTokenizer,
    target_tokenizer: SentencePieceTokenizer,
) -> TranslationDataset:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Naməlum dataset split: {split}")

    return TranslationDataset(
        source_path=config.processed_data_dir / f"{split}.en",
        target_path=config.processed_data_dir / f"{split}.es",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=config.max_length,
    )


def build_dataloader(
    dataset: TranslationDataset,
    config: TrainingConfig,
    shuffle: bool,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        collate_fn=collate_translation_batch,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=config.num_workers > 0,
    )


def build_seq2seq_model(
    config: TrainingConfig,
    source_vocab_size: int,
    target_vocab_size: int,
) -> Seq2Seq:
    encoder_output_size = config.hidden_size * (2 if config.bidirectional else 1)

    encoder = Encoder(
        vocab_size=source_vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        padding_idx=PAD_ID,
        bidirectional=config.bidirectional,
    )

    decoder = Decoder(
        vocab_size=target_vocab_size,
        embedding_dim=config.embedding_dim,
        encoder_output_size=encoder_output_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        padding_idx=PAD_ID,
        attention_size=config.hidden_size,
    )

    return Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        target_vocab_size=target_vocab_size,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="English-Spanish Seq2Seq modelini train et."
    )
    parser.add_argument(
        "--fast-dev-run",
        action="store_true",
        help="Bir train və validation batch ilə pipeline testi.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="W&B məlumatlarını lokal offline rejimdə saxla.",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Training-i verilmiş checkpoint-dən davam etdir.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig()

    pl.seed_everything(config.seed, workers=True)

    source_tokenizer, target_tokenizer = build_tokenizers(config)

    train_dataset = build_dataset(
        config=config,
        split="train",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )
    validation_dataset = build_dataset(
        config=config,
        split="val",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    train_dataloader = build_dataloader(
        dataset=train_dataset,
        config=config,
        shuffle=True,
    )
    validation_dataloader = build_dataloader(
        dataset=validation_dataset,
        config=config,
        shuffle=False,
    )

    seq2seq_model = build_seq2seq_model(
        config=config,
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
    )

    lightning_module = TranslationLightningModule(
        model=seq2seq_model,
        learning_rate=config.learning_rate,
        teacher_forcing_ratio=config.teacher_forcing_ratio,
        pad_id=PAD_ID,
    )

    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = []
    logger: WandbLogger | bool

    if arguments.fast_dev_run:
        logger = False
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=config.checkpoint_dir,
            filename="epoch-{epoch:02d}-val_loss-{val_loss:.4f}",
            monitor="val_loss",
            mode="min",
            save_top_k=1,
            save_last=True,
            auto_insert_metric_name=False,
        )

        early_stopping_callback = EarlyStopping(
            monitor="val_loss",
            mode="min",
            patience=config.early_stopping_patience,
            min_delta=0.0,
            verbose=True,
        )

        learning_rate_callback = LearningRateMonitor(
            logging_interval="epoch",
        )

        callbacks = [
            checkpoint_callback,
            early_stopping_callback,
            learning_rate_callback,
        ]

        logger = WandbLogger(
            project=config.wandb_project,
            name=config.wandb_run_name,
            offline=arguments.offline,
            log_model=not arguments.offline,
        )
        logger.experiment.config.update(
            serialize_config(config),
            allow_val_change=True,
        )

    trainer = pl.Trainer(
        max_epochs=config.max_epochs,
        accelerator="auto",
        devices=1,
        deterministic=True,
        gradient_clip_val=config.gradient_clip_val,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=config.checkpoint_dir,
        log_every_n_steps=50,
        fast_dev_run=arguments.fast_dev_run,
    )

    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=validation_dataloader,
        ckpt_path=arguments.resume,
    )

    if not arguments.fast_dev_run:
        print(f"Ən yaxşı checkpoint: {checkpoint_callback.best_model_path}")
        print(f"Ən yaxşı val_loss: {checkpoint_callback.best_model_score}")


if __name__ == "__main__":
    main()
