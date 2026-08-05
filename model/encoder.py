import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class Encoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        padding_idx: int,
        bidirectional: bool = True,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size 0-dan böyük olmalıdır.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim 0-dan böyük olmalıdır.")

        if hidden_size <= 0:
            raise ValueError("hidden_size 0-dan böyük olmalıdır.")

        if num_layers <= 0:
            raise ValueError("num_layers 0-dan böyük olmalıdır.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout [0, 1) aralığında olmalıdır.")

        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )

        self.embedding_dropout = nn.Dropout(dropout)

        self.rnn = nn.GRU(
            input_size=embedding_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        if bidirectional:
            self.hidden_bridge: nn.Module = nn.Linear(
                hidden_size * 2,
                hidden_size,
            )
        else:
            self.hidden_bridge = nn.Identity()

    def forward(
        self,
        source_ids: Tensor,
        source_lengths: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if source_ids.ndim != 2:
            raise ValueError("source_ids forması (batch, sequence_length) olmalıdır.")

        if source_lengths.ndim != 1:
            raise ValueError("source_lengths birölçülü tensor olmalıdır.")

        if source_ids.size(0) != source_lengths.size(0):
            raise ValueError("source_ids batch ölçüsü source_lengths ilə uyğun deyil.")

        if torch.any(source_lengths <= 0):
            raise ValueError("Bütün source uzunluqları müsbət olmalıdır.")

        if torch.any(source_lengths > source_ids.size(1)):
            raise ValueError("source_lengths sequence dimension-dan böyük ola bilməz.")

        embedded = self.embedding(source_ids)
        embedded = self.embedding_dropout(embedded)

        packed_embeddings = pack_padded_sequence(
            embedded,
            lengths=source_lengths.cpu(),
            batch_first=True,
            enforce_sorted=False,
        )

        packed_outputs, hidden = self.rnn(packed_embeddings)

        encoder_outputs, _ = pad_packed_sequence(
            packed_outputs,
            batch_first=True,
            total_length=source_ids.size(1),
        )

        batch_size = source_ids.size(0)

        hidden = hidden.reshape(
            self.num_layers,
            self.num_directions,
            batch_size,
            self.hidden_size,
        )

        hidden = hidden.permute(0, 2, 1, 3)
        hidden = hidden.reshape(
            self.num_layers,
            batch_size,
            self.hidden_size * self.num_directions,
        )

        decoder_initial_hidden = torch.tanh(self.hidden_bridge(hidden))

        return encoder_outputs, decoder_initial_hidden


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.collate import collate_translation_batch
    from data.dataset import TranslationDataset
    from data.tokenization import PAD_ID, SentencePieceTokenizer
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

    encoder = Encoder(
        vocab_size=source_tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        padding_idx=PAD_ID,
        bidirectional=config.bidirectional,
    )

    encoder_outputs, decoder_hidden = encoder(
        source_ids=batch["src_ids"],
        source_lengths=batch["src_lengths"],
    )

    print(f"Encoder outputs: {encoder_outputs.shape}")
    print(f"Decoder initial hidden: {decoder_hidden.shape}")
