"""Differential WER (dWER) (see https://arxiv.org/abs/1911.07953).

Authors
 * Luca Della Libera 2024
"""

import os

import torch
import torchaudio
from speechbrain.decoders.seq2seq import S2SWhisperGreedySearch
from speechbrain.lobes.models.huggingface_transformers import Whisper
from speechbrain.utils.metric_stats import ErrorRateStats


__all__ = ["DWER"]


SAMPLING_RATE = 16000


class ComputeScore:
    def __init__(self, model_hub, save_path, sampling_rate):
        self.model = Whisper(
            model_hub,
            save_path,
            sampling_rate,
            freeze=True,
            freeze_encoder=True,
        )
        self.searcher = S2SWhisperGreedySearch(
            self.model,
            bos_index=50363,
            eos_index=50257,
            min_decode_ratio=0.0,
            max_decode_ratio=1.0,
        )
        self.model.tokenizer.set_prefix_tokens("english", "transcribe", False)
        self.searcher.set_decoder_input_tokens(
            self.model.tokenizer.prefix_tokens
        )
        self.searcher.set_language_token(self.model.tokenizer.prefix_tokens[1])
        self.wer_computer = ErrorRateStats()
        self.avg_wer_computer = ErrorRateStats()

    def __call__(self, hyp_audio, ref_audio, sampling_rate):
        # Resample
        hyp_audio = torchaudio.functional.resample(
            hyp_audio, sampling_rate, self.model.sampling_rate
        )
        ref_audio = torchaudio.functional.resample(
            ref_audio, sampling_rate, self.model.sampling_rate
        )

        # Forward
        self.model.to(hyp_audio.device)
        self.model.eval()

        max_length = max(len(hyp_audio), len(ref_audio))
        hyp_audio = torch.nn.functional.pad(
            hyp_audio, [0, max_length - len(hyp_audio)]
        )
        ref_audio = torch.nn.functional.pad(
            ref_audio, [0, max_length - len(ref_audio)]
        )

        enc_out = self.model.forward_encoder(
            torch.stack([hyp_audio, ref_audio])
        )
        text, _, _, _ = self.searcher(enc_out, torch.as_tensor([1.0, 1.0]))
        text = self.model.tokenizer.batch_decode(text, skip_special_tokens=True)
        hyp_text = [self.model.tokenizer._normalize(text[0]).split(" ")]
        ref_text = [self.model.tokenizer._normalize(text[1]).split(" ")]

        # Compute WER
        self.wer_computer.append(["ID"], hyp_text, ref_text)
        self.avg_wer_computer.append(["ID"], hyp_text, ref_text)
        dwer = self.wer_computer.summarize("WER")
        avg_dwer = self.avg_wer_computer.summarize("WER")
        self.wer_computer.clear()
        return dwer, avg_dwer, " ".join(hyp_text[0]), " ".join(ref_text[0])


root_folder = os.path.dirname(os.path.realpath(__file__))
save_path = os.path.join(root_folder, "huggingface")

DWER = ComputeScore("openai/whisper-small", save_path, SAMPLING_RATE)
