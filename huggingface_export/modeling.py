import torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin


def _cnn_block(in_ch, out_ch):
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    )


class CNN10Hub(nn.Module, PyTorchModelHubMixin):
    """Hub-exportable version of CNN10Model's architecture (weights-compatible)."""

    def __init__(self, num_channels=1, num_labels=2, dropout_rate=0.2, **kwargs):
        super().__init__()
        layers = []
        layers.append(_cnn_block(num_channels, 64))
        layers.append(_cnn_block(64, 64))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(_cnn_block(64, 128))
        layers.append(_cnn_block(128, 128))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(_cnn_block(128, 256))
        layers.append(_cnn_block(256, 256))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(_cnn_block(256, 512))
        layers.append(_cnn_block(512, 512))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(512, 512))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(512, num_labels))

        self.acoustic_model = nn.Sequential(*layers)

    def forward(self, x):
        return self.acoustic_model(x)


class CNN12Hub(nn.Module, PyTorchModelHubMixin):
    """Hub-exportable version of CNN12Model's architecture (weights-compatible)."""

    def __init__(self, num_channels=1, num_labels=2, dropout_rate=0.2, **kwargs):
        super().__init__()
        layers = []
        layers.append(_cnn_block(num_channels, 64))
        layers.append(_cnn_block(64, 64))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(_cnn_block(64, 128))
        layers.append(_cnn_block(128, 128))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(_cnn_block(128, 256))
        layers.append(_cnn_block(256, 256))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(_cnn_block(256, 512))
        layers.append(_cnn_block(512, 512))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(_cnn_block(512, 1024))
        layers.append(_cnn_block(1024, 1024))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(dropout_rate))

        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Flatten())
        layers.append(nn.Linear(1024, 1024))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout_rate))
        layers.append(nn.Linear(1024, num_labels))

        self.acoustic_model = nn.Sequential(*layers)

    def forward(self, x):
        return self.acoustic_model(x)