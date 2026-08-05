from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from data.tokenization import PAD_ID

BatchItem = dict[str, Tensor | str]
CollatedBatch = dict[str, Tensor | list[str]]


def collate_translation_batch(
    batch: Sequence[BatchItem],
) -> CollatedBatch:
    if not batch:
        raise ValueError("Boş batch collate edilə bilməz.")

    source_sequences = [
        item["src_ids"] for item in batch if isinstance(item["src_ids"], Tensor)
    ]
    target_sequences = [
        item["tgt_ids"] for item in batch if isinstance(item["tgt_ids"], Tensor)
    ]

    if len(source_sequences) != len(batch):
        raise TypeError("Bütün src_ids elementləri Tensor olmalıdır.")

    if len(target_sequences) != len(batch):
        raise TypeError("Bütün tgt_ids elementləri Tensor olmalıdır.")

    source_lengths = torch.tensor(
        [sequence.size(0) for sequence in source_sequences],
        dtype=torch.long,
    )
    target_lengths = torch.tensor(
        [sequence.size(0) for sequence in target_sequences],
        dtype=torch.long,
    )

    source_ids = pad_sequence(
        source_sequences,
        batch_first=True,
        padding_value=PAD_ID,
    )
    target_ids = pad_sequence(
        target_sequences,
        batch_first=True,
        padding_value=PAD_ID,
    )

    decoder_inputs = target_ids[:, :-1]
    decoder_targets = target_ids[:, 1:]

    source_mask = source_ids.ne(PAD_ID)
    target_mask = decoder_targets.ne(PAD_ID)

    decoder_lengths = target_lengths - 1

    source_texts = [str(item["src_text"]) for item in batch]
    target_texts = [str(item["tgt_text"]) for item in batch]

    return {
        "src_ids": source_ids,
        "src_lengths": source_lengths,
        "src_mask": source_mask,
        "tgt_ids": target_ids,
        "tgt_lengths": target_lengths,
        "decoder_inputs": decoder_inputs,
        "decoder_targets": decoder_targets,
        "decoder_lengths": decoder_lengths,
        "tgt_mask": target_mask,
        "src_text": source_texts,
        "tgt_text": target_texts,
    }


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.dataset import TranslationDataset
    from data.tokenization import SentencePieceTokenizer
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

    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_translation_batch,
    )

    batch = next(iter(dataloader))

    print(f"src_ids: {batch['src_ids'].shape}")
    print(f"src_lengths: {batch['src_lengths']}")
    print(f"src_mask: {batch['src_mask'].shape}")
    print(f"tgt_ids: {batch['tgt_ids'].shape}")
    print(f"decoder_inputs: {batch['decoder_inputs'].shape}")
    print(f"decoder_targets: {batch['decoder_targets'].shape}")
    print(f"tgt_mask: {batch['tgt_mask'].shape}")
    print(f"Decoder ilk tokenləri: {batch['decoder_inputs'][:, 0]}")
