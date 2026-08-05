import argparse
from pathlib import Path

import sentencepiece as spm

from training.config import TrainingConfig

PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3


class SentencePieceTokenizer:
    def __init__(self, model_path: Path | str) -> None:
        self.model_path = Path(model_path)

        if not self.model_path.is_file():
            raise FileNotFoundError(f"Tokenizer modeli tapılmadı: {self.model_path}")

        self.processor = spm.SentencePieceProcessor(model_file=str(self.model_path))

        self._validate_special_tokens()

    def _validate_special_tokens(self) -> None:
        expected_ids = {
            "pad_id": PAD_ID,
            "unk_id": UNK_ID,
            "bos_id": BOS_ID,
            "eos_id": EOS_ID,
        }

        actual_ids = {
            "pad_id": self.processor.pad_id(),
            "unk_id": self.processor.unk_id(),
            "bos_id": self.processor.bos_id(),
            "eos_id": self.processor.eos_id(),
        }

        for name, expected_id in expected_ids.items():
            if actual_ids[name] != expected_id:
                raise ValueError(
                    f"{name} uyğun deyil: gözlənilən={expected_id}, "
                    f"modeldə={actual_ids[name]}"
                )

    @property
    def vocab_size(self) -> int:
        return self.processor.get_piece_size()

    @property
    def pad_id(self) -> int:
        return PAD_ID

    @property
    def unk_id(self) -> int:
        return UNK_ID

    @property
    def bos_id(self) -> int:
        return BOS_ID

    @property
    def eos_id(self) -> int:
        return EOS_ID

    def encode(
        self,
        text: str,
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        normalized_text = text.strip()

        if not normalized_text:
            raise ValueError("Tokenizasiya üçün mətn boş ola bilməz.")

        return self.processor.encode(
            normalized_text,
            out_type=int,
            add_bos=add_bos,
            add_eos=add_eos,
        )

    def decode(self, token_ids: list[int]) -> str:
        filtered_ids = [
            token_id
            for token_id in token_ids
            if token_id not in {PAD_ID, BOS_ID, EOS_ID}
        ]
        return self.processor.decode(filtered_ids)

    def token_to_id(self, token: str) -> int:
        return self.processor.piece_to_id(token)

    def id_to_token(self, token_id: int) -> str:
        if not 0 <= token_id < self.vocab_size:
            raise ValueError(f"Token ID vocabulary sərhədindən kənardır: {token_id}")

        return self.processor.id_to_piece(token_id)


def train_sentencepiece_model(
    input_path: Path,
    model_prefix: Path,
    vocab_size: int,
) -> None:
    if not input_path.is_file():
        raise FileNotFoundError(f"Training faylı tapılmadı: {input_path}")

    if vocab_size <= 4:
        raise ValueError("vocab_size 4-dən böyük olmalıdır.")

    model_prefix.parent.mkdir(parents=True, exist_ok=True)

    spm.SentencePieceTrainer.train(
        input=str(input_path),
        model_prefix=str(model_prefix),
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=PAD_ID,
        unk_id=UNK_ID,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_piece="<pad>",
        unk_piece="<unk>",
        bos_piece="<bos>",
        eos_piece="<eos>",
        hard_vocab_limit=False,
        shuffle_input_sentence=False,
        num_threads=1,
    )


def train_tokenizers(config: TrainingConfig, force: bool = False) -> None:
    source_input = config.processed_data_dir / "train.en"
    target_input = config.processed_data_dir / "train.es"

    source_prefix = config.tokenizer_dir / "source"
    target_prefix = config.tokenizer_dir / "target"

    expected_files = [
        source_prefix.with_suffix(".model"),
        source_prefix.with_suffix(".vocab"),
        target_prefix.with_suffix(".model"),
        target_prefix.with_suffix(".vocab"),
    ]

    existing_files = [path for path in expected_files if path.exists()]

    if existing_files and not force:
        paths = "\n".join(str(path) for path in existing_files)
        raise FileExistsError(
            "Tokenizer artifact-ləri artıq mövcuddur:\n"
            f"{paths}\n"
            "Yenidən yaratmaq üçün --force istifadə et."
        )

    train_sentencepiece_model(
        input_path=source_input,
        model_prefix=source_prefix,
        vocab_size=config.vocab_size,
    )

    train_sentencepiece_model(
        input_path=target_input,
        model_prefix=target_prefix,
        vocab_size=config.vocab_size,
    )

    source_tokenizer = SentencePieceTokenizer(source_prefix.with_suffix(".model"))
    target_tokenizer = SentencePieceTokenizer(target_prefix.with_suffix(".model"))

    print(f"Source vocabulary: {source_tokenizer.vocab_size}")
    print(f"Target vocabulary: {target_tokenizer.vocab_size}")
    print(f"Tokenizer qovluğu: {config.tokenizer_dir.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="English və Spanish SentencePiece modellərini öyrət."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Mövcud tokenizer artifact-lərini yenidən yarat.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig()
    train_tokenizers(config=config, force=arguments.force)


if __name__ == "__main__":
    main()
