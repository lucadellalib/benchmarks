import subprocess

import numpy as np
import torch
import torchaudio
from scipy.io.wavfile import write
from speechbrain.inference.vocoders import UnitHIFIGAN
from speechbrain.lobes.models.huggingface_transformers.discrete_wavlm import (
    DiscreteWavLM,
)
from speechbrain.lobes.models.huggingface_transformers.encodec import Encodec


def play_waveform(
    waveform, sample_rate, output_file="waveform.wav", interactive=False
):
    """Play a waveform (requires FFplay installed on the system).

    Arguments
    ---------
    waveform : np.ndarray
        The raw waveform, shape: [num_frames].
    sample_rate : int
        The sample rate.
    output_file : str, optional
        The path to the output file.
    interactive : bool, optional
        True to play interactively, False otherwise.

    """
    waveform = np.array(waveform)
    if waveform.ndim == 1:
        waveform = waveform[None]

    write(output_file, sample_rate, np.transpose(waveform))
    if interactive:
        subprocess.call(["ffplay", output_file])


waveform, sr = torchaudio.load(
    "/media/luca/36025e0f-010c-4721-b02c-2687db813bda/luca/MiniLibriMix/Libri2Mix/wav16k/min/train-100/s2/209-157830-0018_4438-52195-0008.wav"
)
waveform = torchaudio.functional.resample(waveform, sr, 16000)

model_hub = "microsoft/wavlm-large"
save_path = "savedir"
ssl_layer_num = 7
kmeans_repo_id = "speechbrain/SSL_Quantization"
kmeans_filename = "LJSpeech_wavlm_k512_L7.pt"
kmeans_cache_dir = "savedir"
model = DiscreteWavLM(
    model_hub,
    save_path,
    freeze=True,
    ssl_layer_num=ssl_layer_num,
    kmeans_repo_id=kmeans_repo_id,
    kmeans_filename=kmeans_filename,
    kmeans_cache_dir=kmeans_cache_dir,
)
embs, tokens = model(waveform)


waveform = torchaudio.functional.resample(waveform, sr, 24000)
codec = Encodec(
    source="facebook/encodec_24khz",    # Only the 24kHz version supports mono audio
    save_path="prova",
    sample_rate=24000,
    bandwidth=1.5,
    flat_embeddings=False,
    freeze=True,
    renorm_embeddings=False,
)
in_tokens, in_embs = codec.encode(
    waveform, torch.as_tensor([1.0])
)



hifi_gan_unit = UnitHIFIGAN.from_hparams(
    source="chaanks/hifigan-unit-wavlm-l7-k512-ljspeech-ljspeech",
    savedir=save_path,
)
codes = tokens[0]
rec_waveform = hifi_gan_unit.decode_unit(codes)

play_waveform(waveform, 16000, "waveform.wav", interactive=True)
play_waveform(rec_waveform, 16000, "rec_waveform.wav", interactive=True)
