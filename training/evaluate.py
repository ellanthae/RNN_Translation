import argparse
from pathlib import Path

import sacrebleu
import torch
from torch import Tensor, nn
from tqdm import tqdm

import wandb
from data.tokenization import PAD_ID
from training.config import TrainingConfig
from training.lightning_module import TranslationLightningModule
from training.train import (
    build_dataloader,
    build_dataset,
    build_seq2seq_model,
    build_tokenizers,
    serialize_config,
)


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Seq2Seq checkpoint-i test dataset üzərində qiymətləndir."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Qiymətləndiriləcək .ckpt faylı.",
    )
    parser.add_argument(
        "--offline",
        action="store_true",
        help="W&B-ni offline rejimdə işlət.",
    )
    parser.add_argument(
        "--limit-batches",
        type=int,
        default=None,
        help="Yalnız verilən sayda batch qiymətləndir.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=10,
        help="Terminal və W&B üçün nümunə tərcümə sayı.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=1,
        help="1 greedy decoding, 1-dən böyük olduqda beam search.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.6,
        help="Beam search uzunluq cəzası.",
    )

    return parser.parse_args()


def require_tensor(
    batch: dict[str, Tensor | list[str]],
    key: str,
) -> Tensor:
    value = batch[key]

    if not isinstance(value, Tensor):
        raise TypeError(f"batch[{key!r}] Tensor olmalıdır.")

    return value


def move_tensor_batch_to_device(
    batch: dict[str, Tensor | list[str]],
    device: torch.device,
) -> dict[str, Tensor | list[str]]:
    result: dict[str, Tensor | list[str]] = {}

    for key, value in batch.items():
        result[key] = value.to(device) if isinstance(value, Tensor) else value

    return result


def evaluate(
    module: TranslationLightningModule,
    dataloader: torch.utils.data.DataLoader,
    target_tokenizer,
    device: torch.device,
    max_length: int,
    limit_batches: int | None,
    beam_size: int,
    length_penalty: float,
) -> tuple[dict[str, float], list[str], list[str], list[str]]:
    loss_function = nn.CrossEntropyLoss(
        ignore_index=PAD_ID,
        reduction="sum",
    )

    total_loss = 0.0
    total_valid_tokens = 0
    total_correct_tokens = 0

    predictions: list[str] = []
    references: list[str] = []
    source_sentences: list[str] = []

    module.eval()

    with torch.inference_mode():
        for batch_index, batch in enumerate(tqdm(dataloader, desc="Evaluating")):
            if limit_batches is not None and batch_index >= limit_batches:
                break

            batch = move_tensor_batch_to_device(batch, device)

            source_ids = require_tensor(batch, "src_ids")
            source_lengths = require_tensor(batch, "src_lengths")
            source_mask = require_tensor(batch, "src_mask")
            decoder_inputs = require_tensor(batch, "decoder_inputs")
            decoder_targets = require_tensor(batch, "decoder_targets")

            # Loss və token accuracy hesablaması.
            logits, _ = module.model(
                source_ids=source_ids,
                source_lengths=source_lengths,
                source_mask=source_mask,
                decoder_inputs=decoder_inputs,
                teacher_forcing_ratio=0.0,
            )

            loss = loss_function(
                logits.reshape(-1, logits.size(-1)),
                decoder_targets.reshape(-1),
            )

            valid_mask = decoder_targets.ne(PAD_ID)
            predicted_tokens = logits.argmax(dim=-1)

            total_loss += loss.item()
            total_valid_tokens += valid_mask.sum().item()
            total_correct_tokens += (
                (predicted_tokens.eq(decoder_targets) & valid_mask).sum().item()
            )

            # beam_size=1 olduqda greedy, daha böyük olduqda Beam Search.
            generated_ids, _ = module.model.beam_search_decode(
                source_ids=source_ids,
                source_lengths=source_lengths,
                source_mask=source_mask,
                max_length=max_length,
                beam_size=beam_size,
                length_penalty=length_penalty,
            )

            for token_ids in generated_ids.cpu().tolist():
                prediction = target_tokenizer.decode(token_ids).strip()
                predictions.append(prediction)

            batch_references = batch["tgt_text"]
            batch_sources = batch["src_text"]

            if not isinstance(batch_references, list):
                raise TypeError("tgt_text list olmalıdır.")

            if not isinstance(batch_sources, list):
                raise TypeError("src_text list olmalıdır.")

            references.extend(str(text) for text in batch_references)
            source_sentences.extend(str(text) for text in batch_sources)

    if not predictions:
        raise ValueError("Heç bir test nümunəsi qiymətləndirilmədi.")

    average_loss = total_loss / max(total_valid_tokens, 1)
    token_accuracy = total_correct_tokens / max(total_valid_tokens, 1)

    bleu_score = sacrebleu.corpus_bleu(
        predictions,
        [references],
    ).score

    chrf_score = sacrebleu.corpus_chrf(
        predictions,
        [references],
    ).score

    metrics = {
        "test_loss": average_loss,
        "test_token_accuracy": token_accuracy,
        "test_bleu": bleu_score,
        "test_chrf": chrf_score,
        "evaluated_sentences": float(len(predictions)),
    }

    return metrics, source_sentences, references, predictions


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig()

    if not arguments.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint tapılmadı: {arguments.checkpoint}")

    if arguments.limit_batches is not None and arguments.limit_batches <= 0:
        raise ValueError("limit_batches 0-dan böyük olmalıdır.")

    if arguments.samples < 0:
        raise ValueError("samples mənfi ola bilməz.")

    if arguments.beam_size < 1:
        raise ValueError("beam-size ən azı 1 olmalıdır.")

    if arguments.length_penalty < 0.0:
        raise ValueError("length-penalty mənfi ola bilməz.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_tokenizer, target_tokenizer = build_tokenizers(config)

    test_dataset = build_dataset(
        config=config,
        split="test",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
    )

    test_dataloader = build_dataloader(
        dataset=test_dataset,
        config=config,
        shuffle=False,
    )

    seq2seq_model = build_seq2seq_model(
        config=config,
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
    )

    module = TranslationLightningModule.load_from_checkpoint(
        checkpoint_path=arguments.checkpoint,
        model=seq2seq_model,
        map_location=device,
    )
    module.to(device)
    module.eval()

    metrics, sources, references, predictions = evaluate(
        module=module,
        dataloader=test_dataloader,
        target_tokenizer=target_tokenizer,
        device=device,
        max_length=config.max_length,
        limit_batches=arguments.limit_batches,
        beam_size=arguments.beam_size,
        length_penalty=arguments.length_penalty,
    )

    print("\nMetrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    sample_count = min(
        arguments.samples,
        len(predictions),
    )

    print("\nSample translations:")
    for index in range(sample_count):
        print(f"\nEN: {sources[index]}")
        print(f"Reference ES: {references[index]}")
        print(f"Predicted ES: {predictions[index]}")

    run_mode = "offline" if arguments.offline else "online"

    with wandb.init(
        project=config.wandb_project,
        name=f"{config.wandb_run_name}-evaluation",
        job_type="evaluation",
        mode=run_mode,
        config={
            **serialize_config(config),
            "decoding": ("greedy" if arguments.beam_size == 1 else "beam-search"),
            "beam_size": arguments.beam_size,
            "length_penalty": arguments.length_penalty,
        },
    ) as run:
        run.log(metrics)

        table = wandb.Table(
            columns=[
                "English",
                "Reference Spanish",
                "Predicted Spanish",
            ]
        )

        for index in range(sample_count):
            table.add_data(
                sources[index],
                references[index],
                predictions[index],
            )

        run.log({"sample_translations": table})


if __name__ == "__main__":
    main()
