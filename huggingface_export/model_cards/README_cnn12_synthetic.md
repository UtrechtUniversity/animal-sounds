---
license: apache-2.0
pipeline_tag: audio-classification
library_name: pytorch
tags:
  - bioacoustics
  - pytorch
  - cnn
  - conservation
  - chimpanzee
  - wildlife-monitoring
---

# Chimpanzee Vocalization CNN12 (trained with synthetic data)

A 12-layer Convolutional Neural Network for detecting chimpanzee vocalizations in bioacoustic recordings. Part of the [animal-sounds](https://github.com/UtrechtUniversity/animal-sounds) pipeline developed at Utrecht University.

## Model description

- **Architecture:** CNN12 — 5 convolutional blocks (64→128→256→512→1024 channels), global average pooling, fully-connected head, 2-class output (chimpanzee vocalization vs. background). One additional convolutional block compared to CNN10.
- **Input:** 3-channel mel spectrogram representation (mel spectrogram → PCEN normalization → per-file z-normalization → delta features)
- **Preprocessing:** recordings resampled to 48,000 Hz, filtered with a Butterworth bandpass filter (100–2,000 Hz)
- **Frame length:** 0.5-second audio frames

This checkpoint was trained **with synthetic data**: real chimpanzee vocalizations and background recordings from Cameroon, augmented with synthetic examples created by mixing Cameroonian chimpanzee calls with Zambian background noise, to improve robustness in unseen acoustic environments.

A version trained **without** synthetic data is also available: [`utrechtuniversity/chimp-vocalization-cnn12-sanctuary`](https://huggingface.co/utrechtuniversity/chimp-vocalization-cnn12-sanctuary).

## Training data

- **Sanctuary recordings:** Mefou Primate Sanctuary, Cameroon (December 2019 – January 2020), publicly available (Zwerts et al., 2024)
- **Synthetic augmentation:** Cameroonian chimpanzee calls mixed with background noise from Chimfunshi Wildlife Orphanage Trust (CWOT), Zambia

## Evaluation

Evaluated at file level on held-out recordings from Chimfunshi Wildlife Orphanage Trust (CWOT), Zambia — a different acoustic environment from the training data, used to assess cross-environment generalization. Two recorders ("13b" and "14a") were evaluated separately.

| Test recording | F1 | Precision | Recall | F1-macro | Precision-macro | Recall-macro | AUC |
|---|---|---|---|---|---|---|---|
| 13b | 0.6829 | 0.6829 | 0.6829 | 0.8364 | 0.8364 | 0.8364 | 0.9905 |
| 14a | 0.7059 | 0.6667 | 0.7500 | 0.8516 | 0.8323 | 0.8734 | 0.9917 |

*Metrics are for the positive (chimpanzee) class unless labeled "-macro" (averaged across both classes).*

## Usage

This model is not integrated with `transformers` — it's a plain PyTorch model exported with [`PyTorchModelHubMixin`](https://huggingface.co/docs/huggingface_hub/package_reference/mixins). Copy the class definition below, then load the trained weights directly from the Hub.

```python
import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def _cnn_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNN12Hub(nn.Module, PyTorchModelHubMixin):
    def __init__(self, num_channels=3, num_labels=2, dropout_rate=0.5, **kwargs):
        super().__init__()
        layers = [
            _cnn_block(num_channels, 64), _cnn_block(64, 64), nn.AvgPool2d(2), nn.Dropout(dropout_rate),
            _cnn_block(64, 128), _cnn_block(128, 128), nn.AvgPool2d(2), nn.Dropout(dropout_rate),
            _cnn_block(128, 256), _cnn_block(256, 256), nn.AvgPool2d(2), nn.Dropout(dropout_rate),
            _cnn_block(256, 512), _cnn_block(512, 512), nn.AvgPool2d(2), nn.Dropout(dropout_rate),
            _cnn_block(512, 1024), _cnn_block(1024, 1024), nn.AvgPool2d(2), nn.Dropout(dropout_rate),
            nn.AdaptiveAvgPool2d(1), nn.Dropout(dropout_rate), nn.Flatten(),
            nn.Linear(1024, 1024), nn.ReLU(), nn.Dropout(dropout_rate),
            nn.Linear(1024, num_labels),
        ]
        self.acoustic_model = nn.Sequential(*layers)

    def forward(self, x):
        return self.acoustic_model(x)


model = CNN12Hub.from_pretrained("utrechtuniversity/chimp-vocalization-cnn12-synthetic")
model.eval()
```

**Important:** the model expects input tensors matching the exact preprocessing pipeline described above (3-channel mel + PCEN + delta features, 48kHz source audio, 0.5s frames). Raw audio or standard mel spectrograms without this preprocessing will produce incorrect predictions.

A standalone `preprocess.py` (dependent only on `librosa`, `numpy`, `scipy` — no need to install the full source package) is included in this repo, reproducing the exact training-time feature extraction:

```python
from huggingface_hub import hf_hub_download
import importlib.util
import torch

path = hf_hub_download("utrechtuniversity/chimp-vocalization-cnn12-synthetic", "preprocess.py")
spec = importlib.util.spec_from_file_location("preprocess", path)
preprocess = importlib.util.module_from_spec(spec)
spec.loader.exec_module(preprocess)

features = preprocess.extract_features("my_recording.wav")   # (3, n_mel, n_frames)
chunks = preprocess.chunk_features(features)                 # (n_chunks, 3, n_mel, 64)

with torch.no_grad():
    probs = torch.softmax(model(torch.from_numpy(chunks).float()), dim=1)
```

See the [feature extraction source](https://github.com/UtrechtUniversity/animal-sounds/blob/main/bioacoustics/feature_extraction/README.md) in the GitHub repository for further background on the pipeline.

## Intended use and limitations

- Trained specifically for chimpanzee vocalization detection at the sanctuary/wildlife-orphanage recording conditions described above. Performance on substantially different recording setups, microphone types, or species is not guaranteed.
- CNN12 adds one additional convolutional block over CNN10; in this evaluation the two architectures perform similarly, with CNN10 showing modestly higher recall on this synthetic-augmented cross-environment test.
- Although developed for chimpanzee vocalizations, the underlying pipeline in the source repository can be adapted to other species given labeled training data.

## License

Code released under [Apache 2.0](https://github.com/UtrechtUniversity/animal-sounds/blob/main/LICENSE.md), matching the source GitHub repository.

## Citation

The chimpanzee vocalization dataset used to train this model:

```
Zwerts, J. A., Treep, J., Kaandorp, C. S., Meewis, F., Koot, A. C., & Kaya, H. (2021).
Introducing a central african primate vocalisation dataset for automated species classification.
arXiv preprint. https://arxiv.org/pdf/2101.10390.pdf

Zwerts, J., Treep, J., Zahedi, P., & Kaandorp, C. (2024).
Central African Primate vocalization bioacoustics dataset: Yoda Data publication platform of Utrecht University.
```

A paper describing the CNN10/CNN12 models and cross-environment evaluation presented here is currently in preparation. This model card will be updated with the correct citation once available — in the meantime, please contact Dr. Joeri A. Zwerts (j.a.zwerts@uu.nl) for citation guidance.

## Links

- Source code and full pipeline: [github.com/UtrechtUniversity/animal-sounds](https://github.com/UtrechtUniversity/animal-sounds)
- Contact: Dr. Joeri A. Zwerts, Assistant Professor — j.a.zwerts@uu.nl
- Technical/repository support: [Research Engineering team, Utrecht University](https://utrechtuniversity.github.io/research-engineering/) — research.engineering@uu.nl
