from typing import Any

import pytorch_lightning as pl
import torch
from torch import Tensor, nn

from model.seq2seq import Seq2Seq

Batch = dict[str, Tensor | list[str]]


class TranslationLightningModule(pl.LightningModule):
    def __init__(
        self,
        model: Seq2Seq,
        learning_rate: float,
        teacher_forcing_ratio: float,
        pad_id: int,
    ) -> None:
        super().__init__()

        if learning_rate <= 0:
            raise ValueError("learning_rate 0-dan böyük olmalıdır.")

        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio [0, 1] aralığında olmalıdır.")

        self.model = model
        self.learning_rate = learning_rate
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.pad_id = pad_id

        self.loss_function = nn.CrossEntropyLoss(
            ignore_index=pad_id,
        )

        self.save_hyperparameters(ignore=["model"])

    @staticmethod
    def _require_tensor(batch: Batch, key: str) -> Tensor:
        value = batch[key]

        if not isinstance(value, Tensor):
            raise TypeError(f"batch[{key!r}] Tensor olmalıdır.")

        return value

    def _forward_batch(
        self,
        batch: Batch,
        teacher_forcing_ratio: float,
    ) -> tuple[Tensor, Tensor]:
        source_ids = self._require_tensor(batch, "src_ids")
        source_lengths = self._require_tensor(batch, "src_lengths")
        source_mask = self._require_tensor(batch, "src_mask")
        decoder_inputs = self._require_tensor(batch, "decoder_inputs")
        decoder_targets = self._require_tensor(batch, "decoder_targets")

        logits, _ = self.model(
            source_ids=source_ids,
            source_lengths=source_lengths,
            source_mask=source_mask,
            decoder_inputs=decoder_inputs,
            teacher_forcing_ratio=teacher_forcing_ratio,
        )

        return logits, decoder_targets

    def _calculate_loss(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        return self.loss_function(
            logits.reshape(-1, logits.size(-1)),
            targets.reshape(-1),
        )

    def _calculate_token_accuracy(
        self,
        logits: Tensor,
        targets: Tensor,
    ) -> Tensor:
        predictions = logits.argmax(dim=-1)
        valid_mask = targets.ne(self.pad_id)

        correct_tokens = predictions.eq(targets) & valid_mask
        correct_count = correct_tokens.sum()
        valid_count = valid_mask.sum().clamp_min(1)

        return correct_count.float() / valid_count.float()

    def training_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Tensor:
        del batch_idx

        logits, targets = self._forward_batch(
            batch,
            teacher_forcing_ratio=self.teacher_forcing_ratio,
        )

        loss = self._calculate_loss(logits, targets)
        accuracy = self._calculate_token_accuracy(logits, targets)

        batch_size = targets.size(0)

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "train_token_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def validation_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Tensor:
        del batch_idx

        logits, targets = self._forward_batch(
            batch,
            teacher_forcing_ratio=0.0,
        )

        loss = self._calculate_loss(logits, targets)
        accuracy = self._calculate_token_accuracy(logits, targets)

        batch_size = targets.size(0)

        self.log(
            "val_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log(
            "val_token_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            batch_size=batch_size,
        )

        return loss

    def test_step(
        self,
        batch: Batch,
        batch_idx: int,
    ) -> Tensor:
        del batch_idx

        logits, targets = self._forward_batch(
            batch,
            teacher_forcing_ratio=0.0,
        )

        loss = self._calculate_loss(logits, targets)
        accuracy = self._calculate_token_accuracy(logits, targets)

        batch_size = targets.size(0)

        self.log(
            "test_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )
        self.log(
            "test_token_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=batch_size,
        )

        return loss

    def configure_optimizers(self) -> dict[str, Any]:
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }
