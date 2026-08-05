from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import Dataset

from data.tokenization import EOS_ID, SentencePieceTokenizer


class TranslationDataset(Dataset[dict[str, Tensor | str]]):
    def __init__(
        self,
        source_path: Path | str,
        target_path: Path | str,
        source_tokenizer: SentencePieceTokenizer,
        target_tokenizer: SentencePieceTokenizer,
        max_length: int,
    ) -> None:
        self.source_path = Path(source_path)
        self.target_path = Path(target_path)
        self.source_tokenizer = source_tokenizer
        self.target_tokenizer = target_tokenizer
        self.max_length = max_length

        if self.max_length < 2:
            raise ValueError("max_length ən azı 2 olmalıdır.")

        self.source_sentences = self._read_lines(self.source_path)
        self.target_sentences = self._read_lines(self.target_path)

        if len(self.source_sentences) != len(self.target_sentences):
            raise ValueError(
                "Source və target fayllarında sətir sayı fərqlidir: "
                f"source={len(self.source_sentences)}, "
                f"target={len(self.target_sentences)}"
            )

        if not self.source_sentences:
            raise ValueError("Dataset boşdur.")

    @staticmethod
    def _read_lines(path: Path) -> list[str]:
        if not path.is_file():
            raise FileNotFoundError(f"Dataset faylı tapılmadı: {path}")

        with path.open("r", encoding="utf-8") as file:
            return [line.strip() for line in file if line.strip()]

    def _truncate(self, token_ids: list[int]) -> list[int]:
        if len(token_ids) <= self.max_length:
            return token_ids

        return token_ids[: self.max_length - 1] + [EOS_ID]

    def __len__(self) -> int:
        return len(self.source_sentences)

    def __getitem__(self, index: int) -> dict[str, Tensor | str]:
        source_text = self.source_sentences[index]
        target_text = self.target_sentences[index]

        source_ids = self.source_tokenizer.encode(
            source_text,
            add_bos=True,
            add_eos=True,
        )
        target_ids = self.target_tokenizer.encode(
            target_text,
            add_bos=True,
            add_eos=True,
        )

        source_ids = self._truncate(source_ids)
        target_ids = self._truncate(target_ids)

        return {
            "src_ids": torch.tensor(source_ids, dtype=torch.long),
            "tgt_ids": torch.tensor(target_ids, dtype=torch.long),
            "src_text": source_text,
            "tgt_text": target_text,
        }


if __name__ == "__main__":
    from training.config import TrainingConfig

    config = TrainingConfig()

    source_tokenizer = SentencePieceTokenizer(config.tokenizer_dir / "source.model")
    target_tokenizer = SentencePieceTokenizer(config.tokenizer_dir / "target.model")

    dataset = TranslationDataset(
        source_path=config.processed_data_dir / "train.en",
        target_path=config.processed_data_dir / "train.es",
        source_tokenizer=source_tokenizer,
        target_tokenizer=target_tokenizer,
        max_length=config.max_length,
    )

    sample = dataset[0]

    print(f"Dataset ölçüsü: {len(dataset)}")
    print(f"Source text: {sample['src_text']}")
    print(f"Target text: {sample['tgt_text']}")
    print(f"Source IDs: {sample['src_ids']}")
    print(f"Target IDs: {sample['tgt_ids']}")
    print(f"Source uzunluğu: {len(sample['src_ids'])}")
    print(f"Target uzunluğu: {len(sample['tgt_ids'])}")
