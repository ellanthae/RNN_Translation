import torch
from torch import Tensor, nn

from model.attention import BahdanauAttention


class Decoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        encoder_output_size: int,
        hidden_size: int,
        num_layers: int,
        dropout: float,
        padding_idx: int,
        attention_size: int,
    ) -> None:
        super().__init__()

        if vocab_size <= 0:
            raise ValueError("vocab_size 0-dan böyük olmalıdır.")

        if embedding_dim <= 0:
            raise ValueError("embedding_dim 0-dan böyük olmalıdır.")

        if encoder_output_size <= 0:
            raise ValueError("encoder_output_size 0-dan böyük olmalıdır.")

        if hidden_size <= 0:
            raise ValueError("hidden_size 0-dan böyük olmalıdır.")

        if num_layers <= 0:
            raise ValueError("num_layers 0-dan böyük olmalıdır.")

        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout [0, 1) aralığında olmalıdır.")

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=padding_idx,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        self.attention = BahdanauAttention(
            encoder_output_size=encoder_output_size,
            decoder_hidden_size=hidden_size,
            attention_size=attention_size,
        )

        self.rnn = nn.GRU(
            input_size=embedding_dim + encoder_output_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        combined_output_size = hidden_size + encoder_output_size + embedding_dim

        self.pre_output_projection = nn.Linear(
            combined_output_size,
            hidden_size,
        )
        self.output_projection = nn.Linear(
            hidden_size,
            vocab_size,
        )

    def forward(
        self,
        input_token: Tensor,
        hidden: Tensor,
        encoder_outputs: Tensor,
        source_mask: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        if input_token.ndim == 2 and input_token.size(1) == 1:
            input_token = input_token.squeeze(1)

        if input_token.ndim != 1:
            raise ValueError("input_token forması (batch,) olmalıdır.")

        if hidden.ndim != 3:
            raise ValueError(
                "hidden forması (num_layers, batch, hidden_size) olmalıdır."
            )

        batch_size = input_token.size(0)

        expected_hidden_shape = (
            self.num_layers,
            batch_size,
            self.hidden_size,
        )

        if hidden.shape != expected_hidden_shape:
            raise ValueError(
                f"hidden forması uyğun deyil: "
                f"gözlənilən={expected_hidden_shape}, "
                f"faktiki={tuple(hidden.shape)}"
            )

        if encoder_outputs.size(0) != batch_size:
            raise ValueError(
                "input_token və encoder_outputs batch ölçüləri uyğun deyil."
            )

        embedded = self.embedding(input_token)
        embedded = self.embedding_dropout(embedded)

        context_vector, attention_weights = self.attention(
            decoder_hidden=hidden[-1],
            encoder_outputs=encoder_outputs,
            source_mask=source_mask,
        )

        rnn_input = torch.cat(
            [embedded, context_vector],
            dim=-1,
        ).unsqueeze(1)

        rnn_output, next_hidden = self.rnn(
            rnn_input,
            hidden,
        )

        rnn_output = rnn_output.squeeze(1)

        combined_output = torch.cat(
            [rnn_output, context_vector, embedded],
            dim=-1,
        )

        pre_output = torch.tanh(self.pre_output_projection(combined_output))
        logits = self.output_projection(pre_output)

        return logits, next_hidden, attention_weights


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.collate import collate_translation_batch
    from data.dataset import TranslationDataset
    from data.tokenization import PAD_ID, SentencePieceTokenizer
    from model.encoder import Encoder
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

    encoder_output_size = config.hidden_size * (2 if config.bidirectional else 1)

    encoder = Encoder(
        vocab_size=source_tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        padding_idx=PAD_ID,
        bidirectional=config.bidirectional,
    )

    decoder = Decoder(
        vocab_size=target_tokenizer.vocab_size,
        embedding_dim=config.embedding_dim,
        encoder_output_size=encoder_output_size,
        hidden_size=config.hidden_size,
        num_layers=config.num_layers,
        dropout=config.dropout,
        padding_idx=PAD_ID,
        attention_size=config.hidden_size,
    )

    encoder_outputs, hidden = encoder(
        batch["src_ids"],
        batch["src_lengths"],
    )

    first_decoder_token = batch["decoder_inputs"][:, 0]

    logits, next_hidden, attention_weights = decoder(
        input_token=first_decoder_token,
        hidden=hidden,
        encoder_outputs=encoder_outputs,
        source_mask=batch["src_mask"],
    )

    print(f"Input token: {first_decoder_token.shape}")
    print(f"Logits: {logits.shape}")
    print(f"Next hidden: {next_hidden.shape}")
    print(f"Attention: {attention_weights.shape}")
