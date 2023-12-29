"""Simplified transducer searchers.

Authors
 * Luca Della Libera 2023
"""

# Adapted from:
# https://github.com/speechbrain/benchmarks/blob/d068d6142a29a38f6527b29948b180a1f89a21b4/benchmarks/CL_MASR/whisper/model.py#L188

import torch
import torch.nn.functional as F


__all__ = ["STDecoder", "STSearcher"]


_NUM_SPECIAL_TOKENS = 1


class STDecoder(torch.nn.Module):
    """Simplified transducer decoder.

    Arguments
    ---------
    embedding : torch.nn.Module
        The embedding layer.
    decoder : torch.nn.Module
        The decoder.
    decoder_proj : torch.nn.Module
        The decoder projection layer.
    joiner : torch.nn.Module
        The joiner.
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
            The encoder output, shape: ``[batch_size, seq_length, enc_hidden_size]``.
        tokens_bos : torch.LongTensor
            The decoder BOS tokens, shape: ``[batch_size, seq_length, num_codebooks]``.
        enc_out_lens : torch.Tensor
            The relative lengths of the encoder output, shape: ``[batch_size]``.
        hidden_state : Sequence[torch.Tensor]
            The decoder hidden state, shape of i-th tensor: ``[:, batch_size, ...]``.

        Returns
        -------
        torch.Tensor
            The logits, shape: ``[batch_size, seq_length, num_codebooks, vocab_size]``.

        """
        embs_bos = self.embedding(tokens_bos)
        if self.has_hidden_state:
            dec_out, hidden_state = self.decoder(
                embs_bos, lengths=enc_out_lens, hx=hidden_state
            )
        else:
            dec_out = self.decoder(embs_bos, enc_out_lens)[0]
        dec_out = self.decoder_proj(dec_out)
        join_out = self.joiner(enc_out, dec_out)
        logits = self.head(join_out)
        return logits, hidden_state


class STSearcher(torch.nn.Module):
    """Simplified transducer searcher.

    Arguments
    ---------
    decoder : torch.nn.Module
        A module that receives as input the audio features (shape: ``[batch_size, seq_length, hidden_size]``),
        the transcription tokens (shape: ``[batch_size, seq_length, num_codebooks]``),
        and their relative lengths (shape: ``[batch_size]``),
        and returns the corresponding logits (shape: ``[batch_size, seq_length, num_codebooks, vocab_size]``).
    num_codebooks : int
        The number of codebooks.
    bos_id : int
        The BOS index.
    beam_size : int
        The beam size. Greedy search is used if `beam_size` is 1,
        beam search otherwise.
    lm : torch.nn.Module
        A module that receives as input the transcription tokens (shape: ``[batch_size, seq_length, num_codebooks]``)
        and returns the next log probabilities (shape: ``[batch_size, seq_length, num_codebooks, vocab_size]``)
        (used only if `beam_size` > 1).
    lm_weight : float
        The language model weight
        (used only if `beam_size` > 1 and a language model is provided).
    lm_bos_id : int
        The language model BOS index.
        (used only if `beam_size` > 1 and a language model is provided).

    """

    def __init__(
        self,
        decoder,
        num_codebooks,
        bos_id,
        beam_size=1,
        lm=None,
        lm_weight=0.0,
        lm_bos_id=0,
    ):
        super().__init__()
        self.decoder = decoder
        self.num_codebooks = num_codebooks
        self.bos_id = bos_id
        self.beam_size = beam_size
        self.lm = lm
        self.lm_weight = lm_weight
        self.lm_bos_id = lm_bos_id

    def forward(self, audio_features, audio_features_lens):
        """Generate a transcription.

        Arguments
        ---------
        audio_features : torch.Tensor
            A batch of audio features, shape: ``[batch_size, seq_length, hidden_size]``
        audio_features_lens : torch.Tensor
            The relative lengths of the audio features, shape: ``[batch_size]``.

        Returns
        -------
        torch.LongTensor
            The batch of hypotheses, shape: ``[batch_size, seq_length]``

        """
        hyps, _ = generate(
            self.decoder,
            audio_features,
            audio_features_lens,
            self.num_codebooks,
            self.bos_id,
            self.beam_size,
            self.lm,
            self.lm_weight,
            self.lm_bos_id,
        )
        return hyps


def generate(
    decoder,
    audio_features,
    audio_features_lens,
    num_codebooks,
    bos_id,
    beam_size=1,
    lm=None,
    lm_weight=0.0,
    lm_bos_id=0,
    return_all=False,
):
    """Generate a transcription via greedy or beam search.

    Arguments
    ---------
    decoder : torch.nn.Module
        A module that receives as input the audio features (shape: ``[batch_size, seq_length, hidden_size]``),
        the transcription tokens (shape: ``[batch_size, seq_length, num_codebooks]``),
        and their relative lengths (shape: ``[batch_size]``),
        and returns the corresponding logits (shape: ``[batch_size, seq_length, num_codebooks, vocab_size]``).
    audio_features : torch.Tensor
        A batch of audio features, shape: ``[batch_size, seq_length, hidden_size]``
    audio_features_lens : torch.Tensor
        The relative lengths of the audio features, shape: ``[batch_size]``.
    num_codebooks : int
        The number of codebooks.
    bos_id : int
        The BOS index.
    beam_size : int
        The beam size. Greedy search is used if `beam_size` is 1,
        beam search otherwise.
    lm : torch.nn.Module
        A module that receives as input the transcription tokens (shape: ``[batch_size, seq_length, num_codebooks]``)
        and returns the next log probabilities (shape: ``[batch_size, seq_length, num_codebooks, vocab_size]``)
        (used only if `beam_size` > 1).
    lm_weight : float
        The language model weight
        (used only if `beam_size` > 1 and a language model is provided).
    lm_bos_id : int
        The language model BOS index.
        (used only if `beam_size` > 1 and a language model is provided).
    return_all : bool
        True to return all the hypotheses (`beam_size` for each batch element),
        False to return only the one with the highest score.

    Returns
    -------
    torch.LongTensor
        The batch of hypotheses, shape: ``[batch_size, beam_size, seq_length]``
        if `return_all` is True, ``[batch_size, seq_length]`` otherwise.
    torch.Tensor
        The batch of scores, shape: ``[batch_size, beam_size]``
        if `return_all` is True, ``[batch_size]`` otherwise.

    Raises
    ------
    ValueError:
        If an invalid argument value is given.

    """
    batch_size = audio_features.shape[0]
    max_gen_tokens = audio_features.shape[1]
    hyps = torch.full(
        (batch_size, max_gen_tokens + _NUM_SPECIAL_TOKENS, num_codebooks),
        bos_id,
        dtype=torch.long,
        device=audio_features.device,
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
            lm_bos_id,
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
            alive_audio_features[:, : num_gen_tokens + _NUM_SPECIAL_TOKENS],
            alive_hyps if alive_hidden_states is None else alive_hyps[:, -1:],
            hidden_state=alive_hidden_states,
        )
        # B* x K x C
        logits = logits[:, -1]
        log_probs = logits.log_softmax(dim=-1)
        # B* x K
        log_probs, gen_token_ids = log_probs.max(dim=-1)
        # B*
        alive_scores += log_probs.sum(dim=-1)  # Sum along codebook dimension
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
    lm_bos_id=0,
):
    batch_size = audio_features.shape[0]
    num_codebooks = hyps.shape[-1]
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
                alive_hyps
                if alive_hidden_states is None
                else alive_hyps[:, -1:],
                hidden_state=alive_hidden_states,
            )
        else:
            # N x B* x T x K
            alive_hyps = hyps[
                :, alive_mask, : num_gen_tokens + _NUM_SPECIAL_TOKENS
            ]
            # NB* x T x K
            alive_hyps = alive_hyps.movedim(0, 1).reshape(
                beam_size * alive_batch_size, -1, num_codebooks,
            )
            # NB* x T x K x C
            logits, alive_hidden_states = decoder(
                alive_audio_features[:, : num_gen_tokens + _NUM_SPECIAL_TOKENS],
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
            # Set correct BOS index for the language model
            bos_idx = torch.full(
                (alive_hyps.shape[0],),
                lm_bos_id,
                dtype=torch.long,
                device=alive_hyps.device,
            )
            bos_idx = bos_idx[:, None, None].expand(
                -1, alive_hyps.shape[-2], num_codebooks
            )
            lm_log_probs = lm(
                torch.cat([bos_idx, alive_hyps[:, 1:]], dim=-2),  # NB* x T x K
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
                alive_hidden_states = tuple(
                    x.expand(beam_size, *x.shape)
                    .movedim(0, 1)
                    .reshape(-1, beam_size * hyps.shape[1], *x.shape[2:])
                    for x in alive_hidden_states
                )
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
                -1, alive_batch_size, num_codebooks
            )
            # N x B* x K
            gen_token_ids = gen_token_ids.gather(
                0, alive_hyp_idxes[..., None].expand(-1, -1, num_codebooks)
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
        .reshape(beam_size, batch_size, -1, num_codebooks)
        .movedim(0, 1)
    )
    return final_hyps, final_scores
