import subprocess

import numpy as np
from scipy.io.wavfile import write


def play_waveform(waveform, sample_rate, output_file="waveform.wav", interactive=False):
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



import torch
from speechbrain.inference.vocoders import UnitHIFIGAN
from speechbrain.lobes.models.huggingface_transformers.discrete_wavlm import DiscreteWavLM

import torchaudio


#prompt, sr = torchaudio.load("/media/luca/36025e0f-010c-4721-b02c-2687db813bda/luca/MiniLibriMix/Libri2Mix/wav16k/min/train-100/s1/209-4731-0001_3792-176660-0055.wav")
prompt, sr = torchaudio.load("/media/luca/36025e0f-010c-4721-b02c-2687db813bda/luca/MiniLibriMix/Libri2Mix/wav16k/min/train-100/s2/4331-88349-0014_4438-48513-0010.wav")
#prompt, sr = torchaudio.load("/media/luca/36025e0f-010c-4721-b02c-2687db813bda/luca/MiniLibriMix/Libri2Mix/wav16k/min/train-100/s2/209-157830-0018_4438-52195-0008.wav")
s1, sr = torchaudio.load("/media/luca/36025e0f-010c-4721-b02c-2687db813bda/luca/MiniLibriMix/Libri2Mix/wav16k/min/train-100/s1/209-157830-0018_4438-52195-0008.wav")
s2, sr = torchaudio.load("/media/luca/36025e0f-010c-4721-b02c-2687db813bda/luca/MiniLibriMix/Libri2Mix/wav16k/min/train-100/s2/209-157830-0018_4438-52195-0008.wav")
mix = torch.cat([prompt, torch.zeros(1, 10), s1 + s2, torch.zeros(1, 10), prompt], dim=-1)
#mix = 0.8 * mix1 +  0.2*mix2
mix = torchaudio.functional.resample(mix, sr, 16000)

import torch
inputs = torch.rand([10, 600])
model_hub = "microsoft/wavlm-large"
save_path = "savedir"
ssl_layer_num = 7
kmeans_repo_id = "speechbrain/SSL_Quantization"
kmeans_filename = "LJSpeech_wavlm_k512_L7.pt"
kmeans_cache_dir = "savedir"
model = DiscreteWavLM(model_hub, save_path, freeze=True, ssl_layer_num=ssl_layer_num,
                           kmeans_repo_id=kmeans_repo_id, kmeans_filename=kmeans_filename,
                           kmeans_cache_dir=kmeans_cache_dir)
embs, tokens = model(mix)

hifi_gan_unit = UnitHIFIGAN.from_hparams(source="chaanks/hifigan-unit-wavlm-l7-k512-ljspeech-ljspeech", savedir=save_path)
codes = tokens[0]#torch.randint(0, 99, (100,))
waveform = hifi_gan_unit.decode_unit(codes)

#print(prompt.shape[1] + 10)
hyp = waveform[:, (prompt.shape[1] + 10) * 2 : -(prompt.shape[1] + 10) * 2]
play_waveform(mix[:, (prompt.shape[1] + 10) * 2 : -(prompt.shape[1] + 10) * 2], 16000, interactive=True)
play_waveform(mix, 16000, interactive=True)
play_waveform(hyp, 16000, interactive=True)

from dwer import DWER


ref1 = torchaudio.load("/media/luca/36025e0f-010c-4721-b02c-2687db813bda/luca/MiniLibriMix/Libri2Mix/wav16k/min/train-100/s1/209-157830-0018_4438-52195-0008.wav")[0]
ref2 = torchaudio.load("/media/luca/36025e0f-010c-4721-b02c-2687db813bda/luca/MiniLibriMix/Libri2Mix/wav16k/min/train-100/s2/209-157830-0018_4438-52195-0008.wav")[0]
ref1 = torchaudio.functional.resample(ref1, sr, 16000)
ref2 = torchaudio.functional.resample(ref2, sr, 16000)
dwer, hp, txt = DWER(hyp[0], ref1[0], 16000)
print(dwer)
print(hp)
print(txt)