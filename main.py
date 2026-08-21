import argparse
from pathlib import Path

import torch
from torch import Tensor

from data.prepare_data import normalize_text
from data.tokenization import EOS_ID, PAD_ID
from training.config import TrainingConfig
from training.lightning_module import TranslationLightningModule
from training.train import build_seq2seq_model, build_tokenizers


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="İngilis dilindən ispan dilinə interaktiv tərcümə."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="İstifadə ediləcək Lightning checkpoint faylı.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=None,
        help="Maksimum source və generated sequence uzunluğu.",
    )
    parser.add_argument(
        "--beam-size",
        type=int,
        default=5,
        help="Beam search namizədlərinin sayı.",
    )
    parser.add_argument(
        "--length-penalty",
        type=float,
        default=0.6,
        help="Beam search uzunluq cəzası.",
    )
    return parser.parse_args()


def prepare_source(
    text: str,
    source_tokenizer,
    max_length: int,
    device: torch.device,
) -> tuple[Tensor, Tensor, Tensor]:
    normalized_text = normalize_text(text)

    if not normalized_text:
        raise ValueError("Boş mətn tərcümə edilə bilməz.")

    token_ids = source_tokenizer.encode(
        normalized_text,
        add_bos=True,
        add_eos=True,
    )

    if len(token_ids) > max_length:
        print(
            f"Warning: input {len(token_ids)} tokendir; {max_length} tokenə qısaldılır."
        )
        token_ids = token_ids[: max_length - 1] + [EOS_ID]

    source_ids = torch.tensor(
        [token_ids],
        dtype=torch.long,
        device=device,
    )
    source_lengths = torch.tensor(
        [len(token_ids)],
        dtype=torch.long,
        device=device,
    )
    source_mask = source_ids.ne(PAD_ID)

    return source_ids, source_lengths, source_mask


def translate(
    text: str,
    module: TranslationLightningModule,
    source_tokenizer,
    target_tokenizer,
    max_length: int,
    device: torch.device,
    beam_size: int,
    length_penalty: float,
) -> str:
    source_ids, source_lengths, source_mask = prepare_source(
        text=text,
        source_tokenizer=source_tokenizer,
        max_length=max_length,
        device=device,
    )

    generated_ids, _ = module.model.beam_search_decode(
        source_ids=source_ids,
        source_lengths=source_lengths,
        source_mask=source_mask,
        max_length=max_length,
        beam_size=beam_size,
        length_penalty=length_penalty,
    )

    return target_tokenizer.decode(generated_ids[0].cpu().tolist()).strip()


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig()

    if not arguments.ckpt.is_file():
        raise FileNotFoundError(f"Checkpoint tapılmadı: {arguments.ckpt}")

    max_length = arguments.max_length or config.max_length

    if max_length < 2:
        raise ValueError("max_length ən azı 2 olmalıdır.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    source_tokenizer, target_tokenizer = build_tokenizers(config)

    seq2seq_model = build_seq2seq_model(
        config=config,
        source_vocab_size=source_tokenizer.vocab_size,
        target_vocab_size=target_tokenizer.vocab_size,
    )

    if arguments.beam_size < 1:
        raise ValueError("beam-size ən azı 1 olmalıdır.")

    if arguments.length_penalty < 0.0:
        raise ValueError("length-penalty mənfi ola bilməz.")

    module = TranslationLightningModule.load_from_checkpoint(
        checkpoint_path=arguments.ckpt,
        model=seq2seq_model,
        map_location=device,
    )
    module.to(device)
    module.eval()

    print(f"Device: {device}")
    print("Çıxmaq üçün 'exit' yazın.")

    while True:
        try:
            text = input("\nEnter English text (or 'exit'): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nProgram bağlandı.")
            break

        if text.lower() == "exit":
            print("Program bağlandı.")
            break

        if not text:
            print("Boş mətn daxil etməyin.")
            continue

        spanish_translation = translate(
            text=text,
            module=module,
            source_tokenizer=source_tokenizer,
            target_tokenizer=target_tokenizer,
            max_length=max_length,
            device=device,
            beam_size=arguments.beam_size,
            length_penalty=arguments.length_penalty,
        )

        if spanish_translation:
            print(f"Spanish: {spanish_translation}")
        else:
            print("Spanish: [Model boş nəticə yaratdı]")


if __name__ == "__main__":
    main()
