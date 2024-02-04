"""Simplified transducer implementation.

Authors
 * Luca Della Libera 2023
"""

import torch
import torch.nn.functional as F


__all__ = [
    "SimplifiedTransducerDecoder",
    "SimplifiedTransducerEncoder",
    "SimplifiedTransducerSearcher",
]


_NUM_SPECIAL_TOKENS = 1


class SimplifiedTransducerEncoder(torch.nn.Module):
    """Simplified transducer encoder.

    Arguments
    ---------
    embedding : torch.nn.Embedding
        The embedding layer.
    encoder : torch.nn.Module
        The encoder module.
    encoder_proj : torch.nn.Module
        The encoder projection layer.
    head : torch.nn.Module
        The logits head.

    """

    def __init__(
        self, embedding, encoder, encoder_proj=None, head=None,
    ):
        super().__init__()
        self.embedding = embedding
        self.encoder = encoder
        self.encoder_proj = encoder_proj
        self.head = head

    def forward(self, tokens, tokens_lens=None):
        """Forward pass.

        Arguments
        ---------
        tokens : torch.LongTensor
            The encoder tokens, shape: ``[batch_size, seq_length, num_channels]``.
        tokens_lens : torch.Tensor
            The relative lengths of the encoder tokens, shape: ``[batch_size]``.

        Returns
        -------
        torch.Tensor
            If `head` is provided, the logits, shape: ``[batch_size, seq_length, num_channels, vocab_size]``,
            else the encoder hidden states, shape: ``[batch_size, seq_length, d_model]``.

        """
        batch_size = tokens.shape[0]
        num_channels = tokens.shape[-1]
        vocab_size = self.embedding.num_embeddings // num_channels

        # Offset to select embeddings from the correct channel
        tokens = tokens + torch.arange(  # Copy to avoid side effects
            0, num_channels * vocab_size, vocab_size, device=tokens.device,
        )

        # Forward embedding layer (one for each channel)
        embs = (
            self.embedding(tokens)
            .reshape(batch_size, tokens.shape[1], -1, num_channels,)
            .sum(dim=-1)
        )

        # Forward encoder
        enc_out = self.encoder.encode(embs, tokens_lens)
        if self.encoder_proj is not None:
            enc_out = self.encoder_proj(enc_out)

        # If no head is provided, return encoder output
        if self.head is None:
            return enc_out

        # Compute cross-entropy logits (one for each channel)
        logits = self.head(enc_out)
        num_heads = (
            logits.shape[-1] // vocab_size
        )  # Might have more heads than channels
        logits = logits.reshape(batch_size, -1, num_heads, vocab_size)

        return logits


class SimplifiedTransducerDecoder(torch.nn.Module):
    """Simplified transducer decoder.

    Arguments
    ---------
    embedding : torch.nn.Embedding
        The embedding layer.
    decoder : torch.nn.Module
        The decoder module.
    decoder_proj : torch.nn.Module
        The decoder projection layer.
    joiner : torch.nn.Module
        The joiner module.
    head : torch.nn.Module
        The logits head.
    has_hidden_state : bool
        True if `decoder` has a hidden state (e.g. LSTM), False otherwise.
        In that case, a more efficient implementation is used to speed up inference
        (~2x speed up).

    Warnings
    --------
    The implementation for ``has_hidden_state=True`` is not fully tested,
    so results for the case ``has_hidden_state=True`` might differ from
    the case ``has_hidden_state=False``.

    """

    def __init__(
        self,
        embedding,
        decoder,
        decoder_proj,
        joiner,
        head,
        has_hidden_state=False,
    ):
        super().__init__()
        self.embedding = embedding
        self.decoder = decoder
        self.decoder_proj = decoder_proj
        self.joiner = joiner
        self.head = head
        self.has_hidden_state = has_hidden_state

    def forward(
        self, enc_out, tokens_bos, enc_out_lens=None, hidden_state=None
    ):
        """Forward pass.

        Arguments
        ---------
        enc_out : torch.Tensor
            The encoder hidden states, shape: ``[batch_size, seq_length, d_model]``.
        tokens_bos : torch.LongTensor
            The decoder BOS tokens, shape: ``[batch_size, seq_length, num_channels]``.
        enc_out_lens : torch.Tensor
            The relative lengths of the encoder hidden states, shape: ``[batch_size]``.
        hidden_state : Sequence[torch.Tensor]
            The decoder hidden state, shape of the i-th tensor: ``[:, batch_size, ...]``.

        Returns
        -------
        torch.Tensor
            The logits, shape: ``[batch_size, seq_length, num_channels, vocab_size]``.
        Sequence[torch.Tensor]
            The decoder hidden state, shape of the i-th tensor: ``[:, batch_size, ...]``.

        """
        batch_size = tokens_bos.shape[0]
        num_channels = tokens_bos.shape[-1]
        vocab_size = self.embedding.num_embeddings // num_channels

        # Offset to select embeddings from the correct channel
        tokens_bos = tokens_bos + torch.arange(  # Copy to avoid side effects
            0, num_channels * vocab_size, vocab_size, device=tokens_bos.device,
        )

        # Forward embedding layer (one for each channel)
        embs_bos = (
            self.embedding(tokens_bos)
            .reshape(batch_size, tokens_bos.shape[1], -1, num_channels)
            .sum(dim=-1)
        )

        # Forward decoder
        assert enc_out.shape[1] == embs_bos.shape[1]
        if self.has_hidden_state:
            dec_out, hidden_state = self.decoder(
                embs_bos, lengths=enc_out_lens, hx=hidden_state
            )
        else:
            hidden_state = None
            dec_out = self.decoder.encode(embs_bos, enc_out_lens)
        dec_out = self.decoder_proj(dec_out)

        # Forward joiner
        join_out = self.joiner(enc_out, dec_out)

        # Compute cross-entropy logits (one for each channel)
        logits = self.head(join_out).reshape(
            batch_size, join_out.shape[1], num_channels, -1,
        )

        return logits, hidden_state


class SimplifiedTransducerSearcher(torch.nn.Module):
    """Simplified transducer searcher.

    Arguments
    ---------
    st_decoder : SimplifiedTransducerDecoder
        A simplified transducer decoder.
    beam_size : int
        The beam size. Greedy search is used if `beam_size` is 1,
        beam search otherwise.
    lm : torch.nn.Module
        A module that receives as input the transcription tokens (shape: ``[batch_size, seq_length, num_channels]``)
        and the hidden states (shape of the i-th tensor: ``[:, batch_size, ...]``), and returns the next log
        probabilities (shape: ``[batch_size, seq_length, num_channels, vocab_size]``) and hidden states
        (shape of the i-th tensor: ``[:, batch_size, ...]``)
        (used only if `beam_size` > 1).
    lm_weight : float
        The language model weight
        (used only if `beam_size` > 1 and a language model is provided).

    """

    def __init__(
        self, st_decoder, beam_size=1, lm=None, lm_weight=0.0,
    ):
        super().__init__()
        self.st_decoder = st_decoder
        self.beam_size = beam_size
        self.lm = lm
        self.lm_weight = lm_weight

    def forward(self, enc_out, enc_out_lens, bos_id):
        """Generate a transcription.

        Arguments
        ---------
        enc_out : torch.Tensor
            The encoder hidden states, shape: ``[batch_size, seq_length, d_model]``
        enc_out_lens : torch.Tensor
            The relative lengths of the encoder hidden states, shape: ``[batch_size]``.
        bos_id : torch.LongTensor
            The BOS index, shape: ``[batch_size, num_channels]``.

        Returns
        -------
        torch.LongTensor
            The hypotheses, shape: ``[batch_size, seq_length]``.

        """
        hyps, _ = generate(
            self.st_decoder,
            enc_out,
            enc_out_lens,
            bos_id,
            self.beam_size,
            self.lm,
            self.lm_weight,
        )
        return hyps


# Adapted from:
# https://github.com/speechbrain/benchmarks/blob/d068d6142a29a38f6527b29948b180a1f89a21b4/benchmarks/CL_MASR/whisper/model.py#L188
def generate(
    decoder,
    audio_features,
    audio_features_lens,
    bos_id,
    beam_size=1,
    lm=None,
    lm_weight=0.0,
    return_all=False,
):
    max_gen_tokens = audio_features.shape[1]
    hyps = (
        bos_id[:, None]
        .expand(-1, max_gen_tokens + _NUM_SPECIAL_TOKENS, -1)
        .clone()
    )

    if beam_size > 1:
        hyps, scores = _beam_search(
            decoder,
            audio_features,
            audio_features_lens,
            hyps,
            beam_size,
            lm,
            lm_weight,
        )
        if not return_all:
            hyps, scores = hyps[:, 0], scores[:, 0]
    else:
        hyps, scores = _greedy_search(
            decoder, audio_features, audio_features_lens, hyps,
        )
        if return_all:
            hyps, scores = hyps[:, None], scores[:, None]

    return hyps, scores


def _greedy_search(
    decoder,
    audio_features,  # B x S x F
    audio_features_lens,  # B
    hyps,  # B x T x K
):
    batch_size = audio_features.shape[0]
    abs_audio_features_lens = (
        audio_features.shape[1] * audio_features_lens
    ).long()
    max_gen_tokens = audio_features.shape[1]
    num_gen_tokens = 0

    # B
    alive_mask = torch.ones(
        batch_size, dtype=torch.bool, device=audio_features.device
    )
    # B
    scores = torch.zeros(batch_size, device=audio_features.device)
    # Autoregressive loop
    # B* x S x F
    alive_audio_features = audio_features
    # B*
    alive_scores = scores.clone()
    # : x B* x ...
    alive_hidden_states = None
    # Autoregressive loop
    while True:
        # B* x T x K
        alive_hyps = hyps[alive_mask, : num_gen_tokens + _NUM_SPECIAL_TOKENS]
        # B* x T x K x C
        logits, alive_hidden_states = decoder(
            alive_audio_features[
                :,
                (
                    0
                    if alive_hidden_states is None
                    else num_gen_tokens + _NUM_SPECIAL_TOKENS - 1
                ) : num_gen_tokens
                + _NUM_SPECIAL_TOKENS,
            ],
            alive_hyps if alive_hidden_states is None else alive_hyps[:, -1:],
            hidden_state=alive_hidden_states,
        )
        # B* x K x C
        logits = logits[:, -1]
        log_probs = logits.log_softmax(dim=-1)
        # B* x K
        log_probs, gen_token_ids = log_probs.max(dim=-1)
        # B*
        alive_scores += log_probs.sum(dim=-1)  # Sum along channel dimension
        # B* x K
        hyps[alive_mask, num_gen_tokens + _NUM_SPECIAL_TOKENS] = gen_token_ids
        scores[alive_mask] = alive_scores
        num_gen_tokens += 1
        if num_gen_tokens >= max_gen_tokens:
            break
        # B*
        alive_mask_unchanged = (
            num_gen_tokens < abs_audio_features_lens[alive_mask]
        )
        if not alive_mask_unchanged.all():
            alive_mask[alive_mask == True] = alive_mask_unchanged  # noqa: E712
            if not alive_mask.any():
                break
            # B* x S x F
            alive_audio_features = audio_features[alive_mask]
            # B*
            alive_scores = scores[alive_mask]
            if alive_hidden_states is not None:
                # : x B* x ...
                alive_hidden_states = [
                    x[:, alive_mask_unchanged] for x in alive_hidden_states
                ]
    # B x T x K
    hyps = hyps[:, _NUM_SPECIAL_TOKENS : num_gen_tokens + _NUM_SPECIAL_TOKENS]
    return hyps, scores


def _beam_search(
    decoder,
    audio_features,  # B x S x F
    audio_features_lens,  # B
    hyps,  # B x T x K
    beam_size=2,  # N
    lm=None,
    lm_weight=0.0,
):
    batch_size = audio_features.shape[0]
    num_channels = hyps.shape[-1]
    abs_audio_features_lens = (
        audio_features.shape[1] * audio_features_lens
    ).long()
    max_gen_tokens = audio_features.shape[1]
    num_gen_tokens = 0

    # B
    alive_mask = torch.ones(
        batch_size, dtype=torch.bool, device=audio_features.device
    )
    # B
    scores = torch.zeros(batch_size, device=audio_features.device)
    # N x B x T x K
    final_hyps = hyps.expand(beam_size, -1, -1, -1).clone()
    # N x B
    final_scores = torch.zeros(
        beam_size, batch_size, device=audio_features.device
    )
    # B
    final_hyps_count = torch.zeros(
        batch_size, dtype=torch.long, device=audio_features.device
    )
    # : x B* x ...
    alive_hidden_states = None
    lm_hidden_states = None
    # Autoregressive loop
    while True:
        if num_gen_tokens == 0:
            # B* x S x F
            alive_audio_features = audio_features
            # B* x T x K
            alive_hyps = hyps[:, : num_gen_tokens + _NUM_SPECIAL_TOKENS]
            alive_batch_size = alive_hyps.shape[0]
            # B* x T x K x C
            logits, alive_hidden_states = decoder(
                alive_audio_features[:, : num_gen_tokens + _NUM_SPECIAL_TOKENS],
                alive_hyps,
            )
        else:
            # N x B* x T x K
            alive_hyps = hyps[
                :, alive_mask, : num_gen_tokens + _NUM_SPECIAL_TOKENS
            ]
            # NB* x T x K
            alive_hyps = alive_hyps.movedim(0, 1).reshape(
                beam_size * alive_batch_size, -1, num_channels,
            )
            # NB* x T x K x C
            logits, alive_hidden_states = decoder(
                alive_audio_features[
                    :,
                    (
                        0
                        if alive_hidden_states is None
                        else num_gen_tokens + _NUM_SPECIAL_TOKENS - 1
                    ) : num_gen_tokens
                    + _NUM_SPECIAL_TOKENS,
                ],
                alive_hyps
                if alive_hidden_states is None
                else alive_hyps[:, -1:],
                hidden_state=[x.contiguous() for x in alive_hidden_states]
                if alive_hidden_states is not None
                else None,
            )
        # NB* x K x C or B* x K x C (num_gen_tokens=0)
        logits = logits[:, -1]
        log_probs = logits.log_softmax(dim=-1)

        # Language model
        if lm is not None and lm_weight > 0.0:
            lm.eval()
            # Move to correct device
            lm.to(alive_hyps.device)
            lm_log_probs, lm_hidden_states = lm(
                alive_hyps, lm_hidden_states,  # NB* x T x K
            )
            # NB* x K x C or B* x K x C (num_gen_tokens=0)
            lm_log_probs = lm_log_probs[:, -1]
            # NB* x K x C or B* x K x C (num_gen_tokens=0)
            log_probs += lm_weight * lm_log_probs

        if num_gen_tokens == 0:
            # B*
            alive_scores = scores
            # B* x K x C
            alive_scores = alive_scores[:, None, None] + log_probs
            # C x B* x K
            alive_scores = alive_scores.movedim(-1, 0)
            # N x B* x K
            alive_scores, gen_token_ids = alive_scores.topk(beam_size, dim=0)
            # N x B*
            alive_scores = alive_scores.sum(dim=-1)
            # N x B x S x F
            audio_features = audio_features.expand(beam_size, -1, -1, -1)
            # N x B x T x K
            hyps = hyps.expand(beam_size, -1, -1, -1).clone()
            # N x B
            scores = scores.expand(beam_size, -1).clone()
            # NB* x S x F
            alive_audio_features = audio_features.movedim(0, 1).reshape(
                beam_size * hyps.shape[1], -1, alive_audio_features.shape[-1],
            )
            if alive_hidden_states is not None:
                # : x NB* x ...
                alive_hidden_states = [
                    x.expand(beam_size, *x.shape)
                    .movedim(0, 1)
                    .reshape(-1, beam_size * hyps.shape[1], *x.shape[2:])
                    for x in alive_hidden_states
                ]
            if lm_hidden_states is not None:
                lm_hidden_states = [
                    x.expand(beam_size, *x.shape).reshape(-1, *x.shape[1:])
                    for x in lm_hidden_states
                ]
        else:
            # N x B* x K x C
            log_probs = log_probs.reshape(
                alive_batch_size, beam_size, -1, log_probs.shape[-1]
            ).movedim(0, 1)
            # N x B* x K x N
            log_probs, gen_token_ids = log_probs.topk(beam_size, dim=-1)
            # N x B* x N
            alive_scores = alive_scores[:, :, None] + log_probs.sum(dim=-2)
            # N x N x B*
            alive_scores = alive_scores.movedim(1, -1)
            # NN x B*
            alive_scores = alive_scores.reshape(-1, alive_batch_size)
            # N x B*
            alive_scores, alive_hyp_idxes = alive_scores.topk(beam_size, dim=0)
            # N x N x B* x K
            gen_token_ids = gen_token_ids.movedim(-1, 1)
            # NN x B* x K
            gen_token_ids = gen_token_ids.reshape(
                -1, alive_batch_size, num_channels
            )
            # N x B* x K
            gen_token_ids = gen_token_ids.gather(
                0, alive_hyp_idxes[..., None].expand(-1, -1, num_channels)
            )
            # N x B*
            alive_hyp_idxes //= beam_size
            alive_hyp_idxes += torch.arange(
                0,
                alive_batch_size * beam_size,
                beam_size,
                device=alive_hyp_idxes.device,
            )[None]
            # N x B* x T x K
            hyps[
                :, alive_mask, : num_gen_tokens + _NUM_SPECIAL_TOKENS
            ] = alive_hyps[alive_hyp_idxes]
            if alive_hidden_states is not None:
                # NB* x ...
                alive_hidden_states = [
                    x.movedim(0, 1) for x in alive_hidden_states
                ]
                # N x B* x ...
                alive_hidden_states = [
                    x[alive_hyp_idxes] for x in alive_hidden_states
                ]
                # : x NB* x ...
                alive_hidden_states = [
                    x.movedim(0, 1).reshape(-1, *x.shape[2:]).movedim(0, 1)
                    for x in alive_hidden_states
                ]
        # N x B* x K
        hyps[
            :, alive_mask, num_gen_tokens + _NUM_SPECIAL_TOKENS
        ] = gen_token_ids
        # N x B*
        scores[:, alive_mask] = alive_scores
        # N x B*
        num_gen_tokens += 1
        endoftext = (
            num_gen_tokens >= abs_audio_features_lens[alive_mask]
        ).expand(beam_size, -1)
        if endoftext.any() or num_gen_tokens == max_gen_tokens:
            alive_final_hyps = final_hyps[:, alive_mask]
            alive_final_scores = final_scores[:, alive_mask]
            start_idxes = final_hyps_count[alive_mask]
            alive_new_hyps_count = endoftext.sum(dim=0)
            end_idxes = (start_idxes + alive_new_hyps_count).clamp(
                max=beam_size
            )

            idxes_mask = F.one_hot(start_idxes, num_classes=beam_size + 1)
            idxes_mask -= F.one_hot(end_idxes, num_classes=beam_size + 1)
            idxes_mask = idxes_mask.cumsum(dim=-1)[:, :-1].bool().T
            diff_mask = F.one_hot(
                torch.zeros_like(alive_new_hyps_count),
                num_classes=beam_size + 1,
            )
            start_mask = diff_mask - F.one_hot(
                alive_new_hyps_count, num_classes=beam_size + 1
            )
            start_mask = start_mask.cumsum(dim=-1)[:, :-1].bool().T
            diff_mask -= F.one_hot(
                alive_new_hyps_count.min(beam_size - start_idxes),
                num_classes=beam_size + 1,
            )
            diff_mask = diff_mask.cumsum(dim=-1)[:, :-1].bool().T

            alive_final_hyps.movedim(0, 1)[idxes_mask.T] = hyps[
                :, alive_mask
            ].movedim(0, 1)[endoftext.T][diff_mask.T[start_mask.T]]
            alive_final_scores.movedim(0, 1)[idxes_mask.T] = scores[
                :, alive_mask
            ].movedim(0, 1)[endoftext.T][diff_mask.T[start_mask.T]]

            final_hyps[:, alive_mask] = alive_final_hyps
            final_scores[:, alive_mask] = alive_final_scores
            final_hyps_count[alive_mask] = end_idxes

            alive_scores[endoftext] = -float("inf")
            scores[:, alive_mask] = alive_scores
            if num_gen_tokens >= max_gen_tokens:
                break

            # B*
            alive_mask_unchanged = end_idxes < beam_size
            if not alive_mask_unchanged.all():
                alive_mask[
                    alive_mask == True
                ] = alive_mask_unchanged  # noqa: E712
                if not alive_mask.any():
                    break
                # N x B* x S x F
                alive_audio_features = audio_features[:, alive_mask]
                # N x B*
                alive_scores = scores[:, alive_mask]
                alive_batch_size = alive_scores.shape[1]
                # NB* x S x F
                alive_audio_features = alive_audio_features.movedim(
                    0, 1
                ).reshape(
                    beam_size * alive_batch_size,
                    -1,
                    alive_audio_features.shape[-1],
                )
                if alive_hidden_states is not None:
                    # : x NB* x ...
                    alive_hidden_states = [
                        x.reshape(x.shape[0], beam_size, -1, *x.shape[2:])[
                            :, :, alive_mask_unchanged
                        ].reshape(x.shape[0], -1, *x.shape[2:])
                        for x in alive_hidden_states
                    ]
                if lm_hidden_states is not None:
                    # NB* x ...
                    lm_hidden_states = [
                        x.reshape(beam_size, -1, *x.shape[1:])[
                            :, alive_mask_unchanged
                        ].reshape(-1, *x.shape[1:])
                        for x in lm_hidden_states
                    ]
    # N x B x T x K
    final_hyps = final_hyps[
        :, :, _NUM_SPECIAL_TOKENS : num_gen_tokens + _NUM_SPECIAL_TOKENS
    ]
    # B x N x T x K
    final_hyps = final_hyps.movedim(0, 1)
    # B x N
    final_scores = final_scores.movedim(0, 1)
    final_scores, final_score_idxes = final_scores.sort(
        dim=-1, descending=True, stable=True
    )
    final_score_idxes += torch.arange(
        0, batch_size * beam_size, beam_size, device=final_score_idxes.device,
    )[:, None]
    final_hyps = (
        final_hyps.reshape(batch_size * beam_size, -1)[
            final_score_idxes.movedim(0, 1)
        ]
        .reshape(beam_size, batch_size, -1, num_channels)
        .movedim(0, 1)
    )
    return final_hyps, final_scores
