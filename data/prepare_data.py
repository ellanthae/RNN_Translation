import argparse
import json
import random
import re
import unicodedata
from pathlib import Path

from training.config import TrainingConfig

WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text)
    return WHITESPACE_PATTERN.sub(" ", text).strip()


def read_parallel_corpus(
    input_path: Path,
    max_length: int,
    max_length_ratio: float,
) -> list[tuple[str, str]]:
    sentence_pairs: list[tuple[str, str]] = []
    seen_pairs: set[tuple[str, str]] = set()

    with input_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            columns = line.rstrip("\n").split("\t")

            if len(columns) < 2:
                continue

            source_text = normalize_text(columns[0])
            target_text = normalize_text(columns[1])

            if not source_text or not target_text:
                continue

            source_length = len(source_text.split())
            target_length = len(target_text.split())

            if source_length > max_length or target_length > max_length:
                continue

            shorter_length = min(source_length, target_length)
            longer_length = max(source_length, target_length)

            if shorter_length == 0:
                continue

            length_ratio = longer_length / shorter_length
            if length_ratio > max_length_ratio:
                continue

            pair = (source_text, target_text)

            if pair in seen_pairs:
                continue

            seen_pairs.add(pair)
            sentence_pairs.append(pair)

    if not sentence_pairs:
        raise ValueError(
            f"Etibarlı cümlə cütü tapılmadı. Dataset formatını yoxla: {input_path}"
        )

    return sentence_pairs


def split_dataset(
    sentence_pairs: list[tuple[str, str]],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[
    list[tuple[str, str]],
    list[tuple[str, str]],
    list[tuple[str, str]],
]:
    shuffled_pairs = sentence_pairs.copy()
    random.Random(seed).shuffle(shuffled_pairs)

    dataset_size = len(shuffled_pairs)
    train_size = int(dataset_size * train_ratio)
    val_size = int(dataset_size * val_ratio)

    train_end = train_size
    val_end = train_end + val_size

    train_pairs = shuffled_pairs[:train_end]
    val_pairs = shuffled_pairs[train_end:val_end]
    test_pairs = shuffled_pairs[val_end:]

    if not train_pairs or not val_pairs or not test_pairs:
        raise ValueError(
            "Train, validation və ya test split boşdur. "
            "Dataset ölçüsünü və split nisbətlərini yoxla."
        )

    return train_pairs, val_pairs, test_pairs


def write_split(
    sentence_pairs: list[tuple[str, str]],
    output_dir: Path,
    split_name: str,
) -> None:
    source_path = output_dir / f"{split_name}.en"
    target_path = output_dir / f"{split_name}.es"

    with (
        source_path.open("w", encoding="utf-8", newline="\n") as source_file,
        target_path.open("w", encoding="utf-8", newline="\n") as target_file,
    ):
        for source_text, target_text in sentence_pairs:
            source_file.write(f"{source_text}\n")
            target_file.write(f"{target_text}\n")


def write_metadata(
    output_dir: Path,
    input_path: Path,
    train_size: int,
    val_size: int,
    test_size: int,
    config: TrainingConfig,
    max_length_ratio: float,
) -> None:
    metadata = {
        "input_path": str(input_path),
        "seed": config.seed,
        "max_length": config.max_length,
        "max_length_ratio": max_length_ratio,
        "train_ratio": config.train_ratio,
        "val_ratio": config.val_ratio,
        "test_ratio": config.test_ratio,
        "train_size": train_size,
        "val_size": val_size,
        "test_size": test_size,
        "total_size": train_size + val_size + test_size,
    }

    metadata_path = output_dir / "metadata.json"

    with metadata_path.open("w", encoding="utf-8") as file:
        json.dump(metadata, file, ensure_ascii=False, indent=2)


def prepare_data(
    input_path: Path,
    config: TrainingConfig,
    max_length_ratio: float,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Dataset tapılmadı: {input_path}\n"
            "spa.txt faylını artifacts/raw_data/ qovluğuna yerləşdir."
        )

    config.processed_data_dir.mkdir(parents=True, exist_ok=True)

    sentence_pairs = read_parallel_corpus(
        input_path=input_path,
        max_length=config.max_length,
        max_length_ratio=max_length_ratio,
    )

    train_pairs, val_pairs, test_pairs = split_dataset(
        sentence_pairs=sentence_pairs,
        train_ratio=config.train_ratio,
        val_ratio=config.val_ratio,
        seed=config.seed,
    )

    write_split(train_pairs, config.processed_data_dir, "train")
    write_split(val_pairs, config.processed_data_dir, "val")
    write_split(test_pairs, config.processed_data_dir, "test")

    write_metadata(
        output_dir=config.processed_data_dir,
        input_path=input_path,
        train_size=len(train_pairs),
        val_size=len(val_pairs),
        test_size=len(test_pairs),
        config=config,
        max_length_ratio=max_length_ratio,
    )

    print(f"Ümumi cümlə cütü: {len(sentence_pairs)}")
    print(f"Train: {len(train_pairs)}")
    print(f"Validation: {len(val_pairs)}")
    print(f"Test: {len(test_pairs)}")
    print(f"Nəticə qovluğu: {config.processed_data_dir.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="English-Spanish paralel korpusunu hazırla."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Raw paralel corpus faylı. Default: artifacts/raw_data/spa.txt",
    )
    parser.add_argument(
        "--max-length-ratio",
        type=float,
        default=3.0,
        help="İki cümlə arasında icazə verilən maksimum uzunluq nisbəti.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig()

    input_path = arguments.input or config.raw_data_dir / "spa.txt"

    if arguments.max_length_ratio < 1.0:
        raise ValueError("max_length_ratio ən azı 1.0 olmalıdır.")

    prepare_data(
        input_path=input_path,
        config=config,
        max_length_ratio=arguments.max_length_ratio,
    )


if __name__ == "__main__":
    main()
