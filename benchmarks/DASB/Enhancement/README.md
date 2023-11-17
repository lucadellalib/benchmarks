# Speech Enhancement with Discrete Speech Representations

This [SpeechBrain](https://speechbrain.github.io) recipe includes scripts to train speech enhancement systems
based on discrete speech representations.

---------------------------------------------------------------------------------------------------------

## ⚡ Datasets

### VoiceBank

Download the following files from the [official website](https://datashare.ed.ac.uk/handle/10283/2791):

- `clean_testset_wav.zip`
- `clean_trainset_28spk_wav.zip`
- `noisy_testset_wav.zip`
- `noisy_trainset_28spk_wav.zip`

Extract them to a folder of your choice (e.g. `VoiceBank`).

---------------------------------------------------------------------------------------------------------

## 🛠️️ Installation

Open a terminal and run:

```bash
pip install -r requirements.txt
```

---------------------------------------------------------------------------------------------------------

## ▶️ Quickstart

Navigate to `<dataset>/<probing-head>`, open a terminal and run:

```bash
python train_<variant>.py hparams/train_<variant>.yaml --data_folder <path-to-data-folder>
```

### Examples

```bash
cd VoiceBank/Transformer
python train_encodec.py hparams/train_encodec.yaml --data_folder data/VoiceBank
```

---------------------------------------------------------------------------------------------------------

## 📧 Contact

[luca.dellalib@gmail.com](mailto:luca.dellalib@gmail.com)

---------------------------------------------------------------------------------------------------------