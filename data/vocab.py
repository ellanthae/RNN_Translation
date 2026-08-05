import argparse
import json
from pathlib import Path

from data.tokenization import (
    BOS_ID,
    EOS_ID,
    PAD_ID,
    UNK_ID,
    SentencePieceTokenizer,
)
from training.config import TrainingConfig

SPECIAL_TOKENS = {
    "<pad>": PAD_ID,
    "<unk>": UNK_ID,
    "<bos>": BOS_ID,
    "<eos>": EOS_ID,
}


class Vocabulary:
    def __init__(
        self,
        token_to_id: dict[str, int],
        id_to_token: list[str],
    ) -> None:
        self._token_to_id = token_to_id
        self._id_to_token = id_to_token
        self._validate()

    def _validate(self) -> None:
        if len(self._token_to_id) != len(self._id_to_token):
            raise ValueError("token_to_id və id_to_token ölçüləri uyğun deyil.")

        for expected_id, token in enumerate(self._id_to_token):
            actual_id = self._token_to_id.get(token)

            if actual_id != expected_id:
                raise ValueError(
                    f"Uyğunsuz mapping: token={token!r}, "
                    f"gözlənilən ID={expected_id}, faktiki ID={actual_id}"
                )

        for token, expected_id in SPECIAL_TOKENS.items():
            actual_id = self._token_to_id.get(token)

            if actual_id != expected_id:
                raise ValueError(
                    f"Xüsusi token ID-si uyğun deyil: {token}, "
                    f"gözlənilən={expected_id}, faktiki={actual_id}"
                )

    def __len__(self) -> int:
        return len(self._id_to_token)

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

    def token_to_id(self, token: str) -> int:
        return self._token_to_id.get(token, UNK_ID)

    def id_to_token(self, token_id: int) -> str:
        if not 0 <= token_id < len(self):
            return "<unk>"

        return self._id_to_token[token_id]

    def numericalize(self, tokens: list[str]) -> list[int]:
        return [self.token_to_id(token) for token in tokens]

    def denumericalize(self, token_ids: list[int]) -> list[str]:
        return [self.id_to_token(token_id) for token_id in token_ids]

    def save(self, output_path: Path | str) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "token_to_id": self._token_to_id,
            "id_to_token": self._id_to_token,
            "special_tokens": SPECIAL_TOKENS,
            "vocab_size": len(self),
        }

        with output_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, input_path: Path | str) -> "Vocabulary":
        input_path = Path(input_path)

        if not input_path.is_file():
            raise FileNotFoundError(f"Vocabulary faylı tapılmadı: {input_path}")

        with input_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)

        token_to_id = {
            token: int(token_id) for token, token_id in payload["token_to_id"].items()
        }
        id_to_token = list(payload["id_to_token"])

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )

    @classmethod
    def from_tokenizer(
        cls,
        tokenizer: SentencePieceTokenizer,
    ) -> "Vocabulary":
        id_to_token = [
            tokenizer.id_to_token(token_id) for token_id in range(tokenizer.vocab_size)
        ]

        token_to_id = {token: token_id for token_id, token in enumerate(id_to_token)}

        return cls(
            token_to_id=token_to_id,
            id_to_token=id_to_token,
        )


def build_vocabularies(
    config: TrainingConfig,
    force: bool = False,
) -> None:
    source_model_path = config.tokenizer_dir / "source.model"
    target_model_path = config.tokenizer_dir / "target.model"

    source_vocab_path = config.tokenizer_dir / "source_vocab.json"
    target_vocab_path = config.tokenizer_dir / "target_vocab.json"

    existing_files = [
        path for path in [source_vocab_path, target_vocab_path] if path.exists()
    ]

    if existing_files and not force:
        paths = "\n".join(str(path) for path in existing_files)
        raise FileExistsError(
            "Vocabulary artifact-ləri artıq mövcuddur:\n"
            f"{paths}\n"
            "Yenidən yaratmaq üçün --force istifadə et."
        )

    source_tokenizer = SentencePieceTokenizer(source_model_path)
    target_tokenizer = SentencePieceTokenizer(target_model_path)

    source_vocab = Vocabulary.from_tokenizer(source_tokenizer)
    target_vocab = Vocabulary.from_tokenizer(target_tokenizer)

    source_vocab.save(source_vocab_path)
    target_vocab.save(target_vocab_path)

    reloaded_source_vocab = Vocabulary.load(source_vocab_path)
    reloaded_target_vocab = Vocabulary.load(target_vocab_path)

    if len(reloaded_source_vocab) != source_tokenizer.vocab_size:
        raise ValueError("Source vocabulary ölçüsü tokenizer ilə uyğun deyil.")

    if len(reloaded_target_vocab) != target_tokenizer.vocab_size:
        raise ValueError("Target vocabulary ölçüsü tokenizer ilə uyğun deyil.")

    print(f"Source vocabulary: {len(reloaded_source_vocab)}")
    print(f"Target vocabulary: {len(reloaded_target_vocab)}")
    print(f"Vocabulary qovluğu: {config.tokenizer_dir.resolve()}")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SentencePiece vocabulary mapping-lərini hazırla."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Mövcud vocabulary fayllarını yenidən yarat.",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    config = TrainingConfig()
    build_vocabularies(config=config, force=arguments.force)


if __name__ == "__main__":
    main()
