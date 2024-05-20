import torch.nn.functional as F
from torch import nn
import torch


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, hidden_size=2048, channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layers = nn.Sequential(*[self._make_layer(block, 64, nb, stride=1) for nb in num_blocks])
        # self.linear = nn.Linear(512*block.expansion, hidden_size)
        self.linear = nn.Linear(20416, hidden_size)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = torch.flatten(out, start_dim=1)
        out = self.linear(out)
        return out


def resnet18(channels=3, hidden_size=2048):
    return ResNet(BasicBlock, [2], channels=channels, hidden_size=hidden_size)


class RpResNetWork(nn.Module):
    def __init__(self, num_actions, channels, emb_dim=32, hidden_size=128, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.backbone = resnet18(channels=channels, hidden_size=hidden_size)
        self.emb = nn.Embedding(num_actions, emb_dim, max_norm=1)
        self.backbone.fc = nn.Linear(hidden_size, 1)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size+emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, a):
        features = self.backbone(x)
        emb = self.emb(a.squeeze())
        fpe = torch.cat([features, emb], dim=1)
        return self.mlp(fpe)


class ConvNetWork(nn.Module):
    def __init__(self, num_actions, channels, emb_dim=32, hidden_size=128, device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.backbone = nn.Sequential(*[nn.Conv2d(channels, 16, kernel_size=3, stride=1, padding=1, bias=True),
                                        nn.BatchNorm2d(16),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1, bias=True),
                                        nn.BatchNorm2d(32),
                                        nn.ReLU(),
                                        nn.MaxPool2d(2),
                                        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=True),
                                        nn.BatchNorm2d(64),
                                        nn.ReLU()
                                        ])
        self.backbone_fc = nn.Linear(576, hidden_size)
        self.emb = nn.Embedding(num_actions, emb_dim, max_norm=1)

        self.mlp = nn.Sequential(
            nn.Linear(hidden_size+emb_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x, a):
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        features = self.backbone_fc(features)
        emb = self.emb(a.view(-1))
        fpe = torch.cat([features, emb], dim=1)
        return self.mlp(fpe)


class ConvNetWork2(nn.Module):
    def __init__(self, num_actions, channels, emb_dim=32, hidden_sizes=(256, 64), device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=channels,  out_channels=16,  kernel_size=7,stride=3,padding="valid"),
            nn.LeakyReLU(0.01),

            nn.Conv2d(in_channels=16,  out_channels=16,  kernel_size=5,stride=2,padding="valid"),
            nn.LeakyReLU(0.01),
            
            nn.Conv2d(in_channels=16,  out_channels=16,  kernel_size=3,stride=1,padding="valid"),
            nn.LeakyReLU(0.01),

            nn.Conv2d(in_channels=16,  out_channels=16,  kernel_size=3,stride=1,padding="valid"),
            nn.LeakyReLU(0.01)
        )
        self.emb = nn.Embedding(num_actions, emb_dim, max_norm=1)
        self.mlp = nn.Sequential(
            # nn.Linear(2304 + emb_dim, hidden_sizes[0]),
            nn.Linear(784, 64),
            nn.Linear(64, 1),
        )

    def forward(self, x, a):
        features = self.backbone(x)
        features = torch.flatten(features, start_dim=1)
        # emb = self.emb(a.view(-1))
        # fpe = torch.cat([features, emb], dim=1)
        return self.mlp(features).clip(-1, 1)

class DenseNetwork(nn.Module):
    def __init__(self, num_actions, channels, emb_dim=32, hidden_sizes=(256, 64), device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.emb = nn.Embedding(num_actions, emb_dim, max_norm=1)
        self.mlp = nn.Sequential(
            nn.Linear(4050, hidden_sizes[0]),
            nn.Dropout(0.5), 
            nn.BatchNorm1d(hidden_sizes[0]),
            nn.LeakyReLU(0.01),
            
            nn.Linear(hidden_sizes[0], hidden_sizes[1]),
            nn.Dropout(0.5), 
            nn.BatchNorm1d(hidden_sizes[1]),
            nn.LeakyReLU(0.01),
            
            nn.Linear(hidden_sizes[1], 1),
            nn.Tanh()
        )

    def forward(self, x, a):
        features = torch.flatten(x, start_dim=1)
        # emb = self.emb(a.view(-1))
        # fpe = torch.cat([features, emb], dim=1)
        return self.mlp(features)
    
class DenseNetworkTest(nn.Module):
    def __init__(self, num_actions, channels, emb_dim=32, hidden_sizes=(256, 64), device=torch.device("cpu")):
        super().__init__()
        self.device = device
        self.emb = nn.Embedding(num_actions, emb_dim, max_norm=1)
        self.mlp = nn.Sequential(
            nn.Linear(42336, 512),

            nn.LeakyReLU(0.01),
            nn.Linear(512, 256),

            nn.LeakyReLU(0.01),
            
            nn.Linear(256, 64),
            nn.LeakyReLU(0.01),
            
            nn.Linear(64, 1),
            nn.Tanh()
        )

    def forward(self, x, a):
        features = torch.flatten(x, start_dim=1)
        # emb = self.emb(a.view(-1))
        # fpe = torch.cat([features, emb], dim=1)
        return self.mlp(features)