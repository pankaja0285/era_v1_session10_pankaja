import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.channels = {'prep': 64, 'layer1': 128, 'layer2': 256, 'layer3': 512}

        # prep
        self.prep_block = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        
        # layer 1
        self.layer1 = nn.Sequential(
            nn.Conv2d(self.channels['prep'], self.channels['layer1'], 3,
            padding=1, stride=1, bias = False),  # Input: 64x128x3 | Output:62  | RF: 5
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.channels['layer1']),
            nn.ReLU()
        )
        # resblock1
        self.resblock1 = nn.Sequential(
            ## conv block - 1
            nn.Conv2d(self.channels['layer1'], self.channels['layer1'], 3,
            padding=1, stride=1, bias = False), # Input: 128X128X3 | Output:126  | RF: 6
            nn.BatchNorm2d(self.channels['layer1']),
            nn.ReLU(),

            ## conv block - 2
            nn.Conv2d(self.channels['layer1'], self.channels['layer1'], 3,
            padding=1, stride=1, bias = False),  # Input: 128X128X3| Output:  | RF:
            nn.BatchNorm2d(self.channels['layer1']),
            nn.ReLU()
        )

        # layer 2
        self.layer2 = nn.Sequential(
            nn.Conv2d(self.channels['layer1'], self.channels['layer2'], 3,
            padding=1, stride=1, bias = False),  # Input: 128x256x3 | Output:126  | RF: 14
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.channels['layer2']),
            nn.ReLU()
        )

        # layer 3
        self.layer3 = nn.Sequential(
            nn.Conv2d(self.channels['layer2'], self.channels['layer3'], 3,
            padding=1, stride=1, bias = False),  # Input: 256x512x3 | Output:254  | RF: 5
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(self.channels['layer3']),
            nn.ReLU()
        )
        # resblock2
        self.resblock2 = nn.Sequential(
            ## conv block - 4
            nn.Conv2d(self.channels['layer3'], self.channels['layer3'], 3,
            padding=1, stride=1, bias = False), # Input: 512X512X3 | Output:510  | RF: 6
            nn.BatchNorm2d(self.channels['layer3']),
            nn.ReLU(),

            ## conv block - 5
            nn.Conv2d(self.channels['layer3'], self.channels['layer3'], 3,
            padding=1, stride=1, bias = False),  # Input: 512X512X3| Output:510  | RF:
            nn.BatchNorm2d(self.channels['layer3']),
            nn.ReLU()
        )

        self.pool = nn.MaxPool2d(4, 4)
        self.fc1 = nn.Linear(512, 10, bias=False)

    def forward(self, x):
        # Prep Layer
        out = self.prep_block(x)
        # print(f"prep layer shape: {out.shape}")
        out = self.layer1(out)
        # print(f"layer1 layer shape: {out.shape}")

        res1 = self.resblock1(out)
        # print(f"res1 layer shape: {out.shape}")
        out = out + res1
        # print(f"outout after layer1 + res1 shape: {out.shape}")

        out = self.layer2(out)
        # print(f"layer2 layer shape: {out.shape}")

        out = self.layer3(out)
        # print(f"layer3 layer shape: {out.shape}")
        res2 = self.resblock2(out)
        # print(f"res2 layer shape: {out.shape}")
        out = out + res2
        # print(f"outout after layer3 + res2 layer shape: {out.shape}")

        out = self.pool(out)
        # print(f"pool layer shape: {out.shape}")

        out = out.view(out.size(0), -1)
        # print(f"view shape: {out.shape}")

        out = self.fc1(out)
        # print(f"linear shape: {out.shape}")
        return F.log_softmax(out, dim=1)  # out
        