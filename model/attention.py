import torch
from torch import Tensor, nn


class BahdanauAttention(nn.Module):
    def __init__(
        self,
        encoder_output_size: int,
        decoder_hidden_size: int,
        attention_size: int,
    ) -> None:
        super().__init__()

        if encoder_output_size <= 0:
            raise ValueError("encoder_output_size 0-dan böyük olmalıdır.")

        if decoder_hidden_size <= 0:
            raise ValueError("decoder_hidden_size 0-dan böyük olmalıdır.")

        if attention_size <= 0:
            raise ValueError("attention_size 0-dan böyük olmalıdır.")

        self.encoder_projection = nn.Linear(
            encoder_output_size,
            attention_size,
            bias=False,
        )
        self.decoder_projection = nn.Linear(
            decoder_hidden_size,
            attention_size,
            bias=False,
        )
        self.energy_projection = nn.Linear(
            attention_size,
            1,
            bias=False,
        )

    def forward(
        self,
        decoder_hidden: Tensor,
        encoder_outputs: Tensor,
        source_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if decoder_hidden.ndim != 2:
            raise ValueError("decoder_hidden forması (batch, hidden_size) olmalıdır.")

        if encoder_outputs.ndim != 3:
            raise ValueError(
                "encoder_outputs forması "
                "(batch, source_length, encoder_size) olmalıdır."
            )

        if source_mask.ndim != 2:
            raise ValueError("source_mask forması (batch, source_length) olmalıdır.")

        batch_size, source_length, _ = encoder_outputs.shape

        if decoder_hidden.size(0) != batch_size:
            raise ValueError("Decoder və encoder batch ölçüləri uyğun deyil.")

        if source_mask.shape != (batch_size, source_length):
            raise ValueError("source_mask forması encoder_outputs ilə uyğun deyil.")

        if source_mask.dtype != torch.bool:
            raise TypeError("source_mask torch.bool tipində olmalıdır.")

        if not torch.all(source_mask.any(dim=1)):
            raise ValueError("Hər source sequence ən azı bir real token saxlamalıdır.")

        projected_encoder = self.encoder_projection(encoder_outputs)
        projected_decoder = self.decoder_projection(decoder_hidden)
        projected_decoder = projected_decoder.unsqueeze(1)

        energy = torch.tanh(projected_encoder + projected_decoder)
        attention_scores = self.energy_projection(energy).squeeze(-1)

        attention_scores = attention_scores.masked_fill(
            ~source_mask,
            torch.finfo(attention_scores.dtype).min,
        )

        attention_weights = torch.softmax(attention_scores, dim=1)

        context_vector = torch.bmm(
            attention_weights.unsqueeze(1),
            encoder_outputs,
        ).squeeze(1)

        return context_vector, attention_weights


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
        batch["src_ids"],
        batch["src_lengths"],
    )

    encoder_output_size = config.hidden_size * (2 if config.bidirectional else 1)

    attention = BahdanauAttention(
        encoder_output_size=encoder_output_size,
        decoder_hidden_size=config.hidden_size,
        attention_size=config.hidden_size,
    )

    context, weights = attention(
        decoder_hidden=decoder_hidden[-1],
        encoder_outputs=encoder_outputs,
        source_mask=batch["src_mask"],
    )

    print(f"Context: {context.shape}")
    print(f"Attention weights: {weights.shape}")
    print(f"Çəkilərin cəmi: {weights.sum(dim=1)}")

    padded_weights = weights.masked_select(~batch["src_mask"])
    if padded_weights.numel() > 0:
        print(f"PAD üçün maksimum çəki: {padded_weights.max().item()}")
