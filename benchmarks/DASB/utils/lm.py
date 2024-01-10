"""Pretrained EnCodec language model.

Authors
 * Luca Della Libera 2023
"""

import torch

from encodec.model import LMModel, ROOT_URL
from encodec.utils import _get_checkpoint_url


__all__ = ["EncodecLM"]


class EncodecLM(torch.nn.Module):
    """Pretrained EnCodec language model.

    Arguments
    ---------
    variant : str
        The EnCodec variant.

    """

    def __init__(self, variant="encodec_24khz"):
        super().__init__()
        self.model = _get_lm_model(variant)

    def forward(self, tokens_bos, hidden_states=None, offset=0):
        """EnCodec language model.

        Arguments
        ---------
        tokens_bos : torch.LongTensor
            The BOS tokens, shape: ``[batch_size, seq_length, num_codebooks]``.
        hidden_states : torch.Tensor
            The hidden states, shape: ``[...]``.
        offset : int
            The offset.

        Returns
        -------
        torch.Tensor
            The next log probabilities, shape: ``[batch_size, seq_length, num_codebooks, vocab_size]``.
        torch.Tensor
            The hidden states, shape: ``[...]``.
        int
            The offset.

        Warnings
        --------
        The BOS index is hardcoded as 0
        (see https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py#L46)

        """
        tokens_bos = tokens_bos.movedim(-1, -2)
        probs, hidden_states, offset = self.model(
            tokens_bos, hidden_states, offset
        )
        return (
            probs.movedim(-1, -2).movedim(-3, -1).log(),
            hidden_states,
            offset,
        )


# Adapted from:
# https://github.com/facebookresearch/encodec/blob/0e2d0aed29362c8e8f52494baf3e6f99056b214f/encodec/model.py#L199
def _get_lm_model(variant="encodec_24khz") -> "LMModel":
    frame_rates = {
        "encodec_24khz": 75,
        "encodec_48kh": 150,
    }
    checkpoints = {
        "encodec_24khz": "encodec_lm_24khz-1608e3c0.th",
        "encodec_48kh": "encodec_lm_48khz-7add9fc3.th",
    }
    try:
        frame_rate = frame_rates[variant]
        checkpoint_name = checkpoints[variant]
    except KeyError:
        raise RuntimeError("No LM pre-trained for the given EnCodec model")
    url = _get_checkpoint_url(ROOT_URL, checkpoint_name)
    state = torch.hub.load_state_dict_from_url(
        url, map_location="cpu", check_hash=True
    )
    lm = LMModel(
        n_q=32,
        card=1024,
        num_layers=5,
        dim=200,
        past_context=int(3.5 * frame_rate),
    )
    lm.load_state_dict(state)
    lm.eval()
    return lm
