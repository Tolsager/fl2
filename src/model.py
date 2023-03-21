"""ResNet in PyTorch.
For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
"""
import torch.nn as nn
import torch.nn.functional as F
from src import resnet

# from lib.normalize import Normalize


# class BasicBlock(nn.Module):
#     expansion = 1

#     def __init__(self, in_planes, planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=1, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class Bottleneck(nn.Module):
#     expansion = 4

#     def __init__(self, in_planes, planes, stride=1):
#         super(Bottleneck, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(planes)
#         self.conv2 = nn.Conv2d(
#             planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
#         )
#         self.bn2 = nn.BatchNorm2d(planes)
#         self.conv3 = nn.Conv2d(
#             planes, self.expansion * planes, kernel_size=1, bias=False
#         )
#         self.bn3 = nn.BatchNorm2d(self.expansion * planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != self.expansion * planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(
#                     in_planes,
#                     self.expansion * planes,
#                     kernel_size=1,
#                     stride=stride,
#                     bias=False,
#                 ),
#                 nn.BatchNorm2d(self.expansion * planes),
#             )

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = F.relu(self.bn2(self.conv2(out)))
#         out = self.bn3(self.conv3(out))
#         out += self.shortcut(x)
#         out = F.relu(out)
#         return out


# class ResNet(nn.Module):
#     def __init__(self, block, num_blocks):
#         super(ResNet, self).__init__()
#         self.in_planes = 64

#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
#         self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
#         self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
#         self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)

#     def _make_layer(self, block, planes, num_blocks, stride):
#         strides = [stride] + [1] * (num_blocks - 1)
#         layers = []
#         for stride in strides:
#             layers.append(block(self.in_planes, planes, stride))
#             self.in_planes = planes * block.expansion
#         return nn.Sequential(*layers)

#     def forward(self, x):
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = F.avg_pool2d(out, 4)
#         out = out.view(out.size(0), -1)
#         return out


# def ResNet18():
#     return ResNet(BasicBlock, [2, 2, 2, 2])


# def ResNet34():
#     return ResNet(BasicBlock, [3, 4, 6, 3])


# def ResNet50():
#     return ResNet(Bottleneck, [3, 4, 6, 3])


# def ResNet101():
#     return ResNet(Bottleneck, [3, 4, 23, 3])


# def ResNet152():
#     return ResNet(Bottleneck, [3, 8, 36, 3])


class projection_MLP(nn.Module):
    def __init__(self, in_dim, out_dim, num_layers=2):
        super().__init__()
        hidden_dim = out_dim
        self.num_layers = num_layers

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dim, out_dim),
            nn.BatchNorm1d(out_dim, affine=False),  # Page:5, Paragraph:2
        )

    def forward(self, x):
        if self.num_layers == 2:
            x = self.layer1(x)
            x = self.layer3(x)
        elif self.num_layers == 3:
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
        return x


class prediction_MLP(nn.Module):
    def __init__(self, in_dim=2048):
        super().__init__()
        out_dim = in_dim
        hidden_dim = int(out_dim / 4)

        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x


# class SimSiam(nn.Module):
#     def __init__(self, embedding_size: int = 2048):
#         super(SimSiam, self).__init__()
#         self.backbone = ResNet18()
#         self.projector = projection_MLP(512, embedding_size, 2)

#         self.encoder = nn.Sequential(self.backbone, self.projector)

#         self.predictor = prediction_MLP(embedding_size)

#     def forward(self, aug1, aug2):
#         z1 = self.encoder(aug1)
#         z2 = self.encoder(aug2)

#         p1 = self.predictor(z1)
#         p2 = self.predictor(z2)

#         return {"z1": z1, "z2": z2, "p1": p1, "p2": p2}
class SimSiam(nn.Module):
    def __init__(self):
        super(SimSiam, self).__init__()
        self.backbone = SimSiam.get_backbone("resnet18")
        out_dim = self.backbone.fc.weight.shape[1]
        self.backbone.fc = nn.Identity()

        self.projector = projection_MLP(out_dim, 2048, 2)

        self.encoder = nn.Sequential(self.backbone, self.projector)

        self.predictor = prediction_MLP(2048)

    @staticmethod
    def get_backbone(backbone_name):
        return {
            "resnet18": resnet.ResNet18(),
            "resnet34": resnet.ResNet34(),
            "resnet50": resnet.ResNet50(),
            "resnet101": resnet.ResNet101(),
            "resnet152": resnet.ResNet152(),
        }[backbone_name]

    def forward(self, im_aug1, im_aug2):

        z1 = self.encoder(im_aug1)
        z2 = self.encoder(im_aug2)

        p1 = self.predictor(z1)
        p2 = self.predictor(z2)

        return {"z1": z1, "z2": z2, "p1": p1, "p2": p2}
