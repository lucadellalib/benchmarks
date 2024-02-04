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