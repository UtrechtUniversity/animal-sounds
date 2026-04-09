import torch.nn as nn
from bioacoustics.classifier.model.base_torch_model import BaseTorchModel


class CNN12Model(BaseTorchModel):

    num_labels = 2

    def __init__(self, num_channels, output_dir=None, model_dir=None,
                 init_mode="glorot_uniform", dropout_rate=0.2, weight_constraint=None):
        super().__init__(output_dir=output_dir, model_dir=model_dir,
                         num_channels=num_channels,
                         init_mode=init_mode,
                         dropout_rate=dropout_rate, weight_constraint=weight_constraint)

    def _cnn_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _make_cnn_model(self):
        layers = []

        # ---- BLOCK 1 ----
        layers.append(self._cnn_block(self.num_channels,  64))
        layers.append(self._cnn_block(64, 64))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(self.dropout_rate))

        # ---- BLOCK 2 ----
        layers.append(self._cnn_block(64, 128))
        layers.append(self._cnn_block(128, 128))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(self.dropout_rate))

        # ---- BLOCK 3 ----
        layers.append(self._cnn_block(128, 256))
        layers.append(self._cnn_block(256, 256))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(self.dropout_rate))

        # ---- BLOCK 4 ----
        layers.append(self._cnn_block(256, 512))
        layers.append(self._cnn_block(512, 512))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(self.dropout_rate))

        # ---- BLOCK 5 ----
        layers.append(self._cnn_block(512, 1024))
        layers.append(self._cnn_block(1024, 1024))
        layers.append(nn.AvgPool2d(2))
        layers.append(nn.Dropout(self.dropout_rate))

        # Global average pooling
        layers.append(nn.AdaptiveAvgPool2d(1))
        layers.append(nn.Dropout(self.dropout_rate))

        # Fully connected
        layers.append(nn.Flatten())
        layers.append(nn.Linear(1024, 1024))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(self.dropout_rate))

        layers.append(nn.Linear(1024, self.num_labels))

        self.acoustic_model = nn.Sequential(*layers)

    def forward(self, x):
        return self.acoustic_model(x)
