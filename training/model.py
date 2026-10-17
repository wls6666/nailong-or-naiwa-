"""A small CNN trained from scratch. forward returns raw logits."""
from torch import nn


class SmallCNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        layers = []
        for i, (incoming, outgoing) in enumerate([(3, 32), (32, 64), (64, 128), (128, 256)]):
            layers.extend([nn.Conv2d(incoming, outgoing, 3, padding=1),
                           nn.BatchNorm2d(outgoing), nn.ReLU()])
            if i < 3:
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                                  nn.Dropout(0.35), nn.Linear(256, num_classes))

    def encode(self, x):
        # Existing 256D pre-dropout GAP representation; no new parameters.
        return self.head[1](self.head[0](self.features(x)))

    def forward_outputs(self, x):
        embedding = self.encode(x)
        logits = self.head[3](self.head[2](embedding))
        return logits, embedding

    def forward(self, x):
        return self.forward_outputs(x)[0]
