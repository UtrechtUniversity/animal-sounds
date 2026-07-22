# Hugging Face export

This folder contains everything related to publishing the trained CNN10 / CNN12
chimpanzee vocalization classifiers to Hugging Face Hub.

## Published models

| Model | Training data | Hugging Face repo |
|---|---|---|
| CNN10 | Sanctuary + synthetic | [utrechtuniversity/chimp-vocalization-cnn10-synthetic](https://huggingface.co/utrechtuniversity/chimp-vocalization-cnn10-synthetic) |
| CNN10 | Sanctuary only | [utrechtuniversity/chimp-vocalization-cnn10-sanctuary](https://huggingface.co/utrechtuniversity/chimp-vocalization-cnn10-sanctuary) |
| CNN12 | Sanctuary + synthetic | [utrechtuniversity/chimp-vocalization-cnn12-synthetic](https://huggingface.co/utrechtuniversity/chimp-vocalization-cnn12-synthetic) |
| CNN12 | Sanctuary only | [utrechtuniversity/chimp-vocalization-cnn12-sanctuary](https://huggingface.co/utrechtuniversity/chimp-vocalization-cnn12-sanctuary) |

See each repo's model card for architecture details, training data, and cross-environment evaluation results (recorders 13b / 14a, Chimfunshi Wildlife Orphanage Trust, Zambia).

## What's in this folder

- **`modeling.py`** — `CNN10Hub` / `CNN12Hub`, standalone `nn.Module` + `PyTorchModelHubMixin` wrappers matching the architecture of `CNN10Model` / `CNN12Model` in `bioacoustics/classifier/model/`. Uploaded to each Hugging Face repo so `from_pretrained()` works without installing this whole package.
- **`preprocess.py`** — standalone feature extraction (Butterworth bandpass filter → mel spectrogram → PCEN → per-file z-normalization → delta/delta-delta channels), pinned to the exact config values used to train the 4 published checkpoints. Depends only on `librosa`, `numpy`, `scipy` — no need to install this package. Also uploaded to each Hugging Face repo.
- **`model_cards/`** — the four `README.md` files, one per Hugging Face repo.
- **`export_to_hub.py`** — the local script used to load trained `.pth` checkpoints, wrap them with `CNN10Hub`/`CNN12Hub`, and push them (plus `modeling.py`, `preprocess.py`, and the matching model card) to each Hugging Face repo. This script is **not** uploaded to Hugging Face — it's maintenance tooling for re-exporting after retraining, not something end users of the published models need.

## Important: keeping this in sync

`preprocess.py` in this folder hardcodes the exact preprocessing config (`sample_rate`,
`window_length`, `hop_length`, `n_mel`, `low_cut`, `high_cut`, PCEN parameters) used to
train the 4 published checkpoints. Unlike the general, config-driven feature extraction
in `bioacoustics/feature_extraction/`, this is a **frozen snapshot** matching those
specific models.

If the models are retrained with different preprocessing settings, `preprocess.py` (and
the corresponding values documented in each model card) must be updated and re-uploaded
to stay accurate — they do not update automatically.
