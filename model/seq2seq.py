import torch
from torch import Tensor, nn

from model.decoder import Decoder
from model.encoder import Encoder


class Seq2Seq(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        target_vocab_size: int,
        bos_id: int,
        eos_id: int,
        pad_id: int,
    ) -> None:
        super().__init__()

        if target_vocab_size <= 0:
            raise ValueError("target_vocab_size 0-dan böyük olmalıdır.")

        self.encoder = encoder
        self.decoder = decoder
        self.target_vocab_size = target_vocab_size
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.pad_id = pad_id

    def forward(
        self,
        source_ids: Tensor,
        source_lengths: Tensor,
        source_mask: Tensor,
        decoder_inputs: Tensor,
        teacher_forcing_ratio: float,
    ) -> tuple[Tensor, Tensor]:
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio [0, 1] aralığında olmalıdır.")

        if decoder_inputs.ndim != 2:
            raise ValueError("decoder_inputs forması (batch, target_length) olmalıdır.")

        if decoder_inputs.size(1) == 0:
            raise ValueError("decoder_inputs sequence dimension boşdur.")

        if source_ids.size(0) != decoder_inputs.size(0):
            raise ValueError("Source və decoder batch ölçüləri uyğun deyil.")

        encoder_outputs, hidden = self.encoder(
            source_ids,
            source_lengths,
        )

        batch_size, target_length = decoder_inputs.shape

        current_token = decoder_inputs[:, 0]

        logits_per_step: list[Tensor] = []
        attention_per_step: list[Tensor] = []

        for step in range(target_length):
            logits, hidden, attention_weights = self.decoder(
                input_token=current_token,
                hidden=hidden,
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
            )

            logits_per_step.append(logits)
            attention_per_step.append(attention_weights)

            if step + 1 >= target_length:
                continue

            predicted_token = logits.argmax(dim=-1)
            teacher_token = decoder_inputs[:, step + 1]

            if teacher_forcing_ratio == 1.0:
                current_token = teacher_token
            elif teacher_forcing_ratio == 0.0:
                current_token = predicted_token
            else:
                teacher_mask = (
                    torch.rand(batch_size, device=source_ids.device)
                    < teacher_forcing_ratio
                )
                current_token = torch.where(
                    teacher_mask,
                    teacher_token,
                    predicted_token,
                )

        output_logits = torch.stack(logits_per_step, dim=1)
        attention_weights = torch.stack(attention_per_step, dim=1)

        return output_logits, attention_weights

    @torch.no_grad()
    def greedy_decode(
        self,
        source_ids: Tensor,
        source_lengths: Tensor,
        source_mask: Tensor,
        max_length: int,
    ) -> tuple[Tensor, Tensor]:
        if max_length < 2:
            raise ValueError("max_length ən azı 2 olmalıdır.")

        encoder_outputs, hidden = self.encoder(
            source_ids,
            source_lengths,
        )

        batch_size = source_ids.size(0)
        device = source_ids.device

        current_token = torch.full(
            size=(batch_size,),
            fill_value=self.bos_id,
            dtype=torch.long,
            device=device,
        )

        generated_tokens = [current_token]
        attention_per_step: list[Tensor] = []

        finished = torch.zeros(
            batch_size,
            dtype=torch.bool,
            device=device,
        )

        for _ in range(max_length - 1):
            logits, next_hidden, attention_weights = self.decoder(
                input_token=current_token,
                hidden=hidden,
                encoder_outputs=encoder_outputs,
                source_mask=source_mask,
            )

            predicted_token = logits.argmax(dim=-1)

            token_to_store = torch.where(
                finished,
                torch.full_like(predicted_token, self.pad_id),
                predicted_token,
            )

            generated_tokens.append(token_to_store)
            attention_per_step.append(attention_weights)

            newly_finished = predicted_token.eq(self.eos_id)
            finished = finished | newly_finished

            hidden = next_hidden

            if torch.all(finished):
                break

            current_token = torch.where(
                finished,
                torch.full_like(predicted_token, self.pad_id),
                predicted_token,
            )

        generated = torch.stack(generated_tokens, dim=1)

        if attention_per_step:
            attentions = torch.stack(attention_per_step, dim=1)
        else:
            attentions = encoder_outputs.new_empty(
                batch_size,
                0,
                encoder_outputs.size(1),
            )

        return generated, attentions

    @torch.no_grad()
    def beam_search_decode(
        self,
        source_ids: Tensor,
        source_lengths: Tensor,
        source_mask: Tensor,
        max_length: int,
        beam_size: int = 5,
        length_penalty: float = 0.6,
    ) -> tuple[Tensor, Tensor]:
        if max_length < 2:
            raise ValueError("max_length ən azı 2 olmalıdır.")

        if beam_size < 1:
            raise ValueError("beam_size ən azı 1 olmalıdır.")

        if length_penalty < 0.0:
            raise ValueError("length_penalty mənfi ola bilməz.")

        if beam_size == 1:
            return self.greedy_decode(
                source_ids=source_ids,
                source_lengths=source_lengths,
                source_mask=source_mask,
                max_length=max_length,
            )

        encoder_outputs, initial_hidden = self.encoder(
            source_ids,
            source_lengths,
        )

        batch_size = source_ids.size(0)
        device = source_ids.device
        generated_sequences: list[Tensor] = []

        def normalized_score(
            score: float,
            sequence_length: int,
        ) -> float:
            generated_length = max(sequence_length - 1, 1)
            penalty = ((5.0 + generated_length) / 6.0) ** length_penalty
            return score / penalty

        for sample_index in range(batch_size):
            sample_encoder_outputs = encoder_outputs[sample_index : sample_index + 1]
            sample_source_mask = source_mask[sample_index : sample_index + 1]
            sample_hidden = initial_hidden[
                :, sample_index : sample_index + 1
            ].contiguous()

            # (tokenlər, log-score, hidden, tamamlanıb?)
            beams: list[tuple[list[int], float, Tensor, bool]] = [
                ([self.bos_id], 0.0, sample_hidden, False)
            ]

            for _ in range(max_length - 1):
                candidates: list[tuple[list[int], float, Tensor, bool]] = []

                for tokens, score, hidden, finished in beams:
                    if finished:
                        candidates.append((tokens, score, hidden, True))
                        continue

                    input_token = torch.tensor(
                        [tokens[-1]],
                        dtype=torch.long,
                        device=device,
                    )

                    logits, next_hidden, _ = self.decoder(
                        input_token=input_token,
                        hidden=hidden,
                        encoder_outputs=sample_encoder_outputs,
                        source_mask=sample_source_mask,
                    )

                    log_probabilities = torch.log_softmax(
                        logits,
                        dim=-1,
                    ).squeeze(0)

                    # BOS və PAD normal generated token olmamalıdır.
                    log_probabilities[self.bos_id] = float("-inf")
                    log_probabilities[self.pad_id] = float("-inf")

                    # Eyni tokenin ardıcıl iki dəfə yaradılmasını blokla.
                    previous_token = tokens[-1]

                    if previous_token not in {
                        self.bos_id,
                        self.eos_id,
                        self.pad_id,
                    }:
                        log_probabilities[previous_token] = float("-inf")

                    top_scores, top_tokens = torch.topk(
                        log_probabilities,
                        k=beam_size,
                    )

                    for token_score, token_id in zip(
                        top_scores.tolist(),
                        top_tokens.tolist(),
                        strict=True,
                    ):
                        next_tokens = [*tokens, token_id]
                        next_score = score + token_score
                        next_finished = token_id == self.eos_id

                        candidates.append(
                            (
                                next_tokens,
                                next_score,
                                next_hidden,
                                next_finished,
                            )
                        )

                candidates.sort(
                    key=lambda candidate: normalized_score(
                        candidate[1],
                        len(candidate[0]),
                    ),
                    reverse=True,
                )
                beams = candidates[:beam_size]

                if all(beam[3] for beam in beams):
                    break

            finished_beams = [beam for beam in beams if beam[3]]
            selectable_beams = finished_beams or beams

            best_tokens, _, _, _ = max(
                selectable_beams,
                key=lambda candidate: normalized_score(
                    candidate[1],
                    len(candidate[0]),
                ),
            )

            generated_sequences.append(
                torch.tensor(
                    best_tokens,
                    dtype=torch.long,
                    device=device,
                )
            )

        output_length = max(sequence.size(0) for sequence in generated_sequences)

        generated = torch.full(
            (batch_size, output_length),
            fill_value=self.pad_id,
            dtype=torch.long,
            device=device,
        )

        for index, sequence in enumerate(generated_sequences):
            generated[index, : sequence.size(0)] = sequence

        empty_attention = encoder_outputs.new_empty(
            batch_size,
            0,
            encoder_outputs.size(1),
        )

        return generated, empty_attention


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    from data.collate import collate_translation_batch
    from data.dataset import TranslationDataset
    from data.tokenization import (
        BOS_ID,
        EOS_ID,
        PAD_ID,
        SentencePieceTokenizer,
    )
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

    model = Seq2Seq(
        encoder=encoder,
        decoder=decoder,
        target_vocab_size=target_tokenizer.vocab_size,
        bos_id=BOS_ID,
        eos_id=EOS_ID,
        pad_id=PAD_ID,
    )

    logits, train_attention = model(
        source_ids=batch["src_ids"],
        source_lengths=batch["src_lengths"],
        source_mask=batch["src_mask"],
        decoder_inputs=batch["decoder_inputs"],
        teacher_forcing_ratio=config.teacher_forcing_ratio,
    )

    model.eval()

    generated, inference_attention = model.greedy_decode(
        source_ids=batch["src_ids"],
        source_lengths=batch["src_lengths"],
        source_mask=batch["src_mask"],
        max_length=config.max_length,
    )

    print(f"Training logits: {logits.shape}")
    print(f"Training attention: {train_attention.shape}")
    print(f"Generated IDs: {generated.shape}")
    print(f"Inference attention: {inference_attention.shape}")
    print(f"Generated ilk token: {generated[:, 0]}")
