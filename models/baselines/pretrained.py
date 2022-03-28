from torch import nn
from torchvision.models import vgg11, densenet121, resnet34, inception_v3
import numpy as np
import torch.nn.functional as F
from modules.utils import Normalise


def convert_lin2conv2d(lin_mod, ks=1):
    shape = np.array(lin_mod.weight.shape )
    shape[1] = shape[1]//(ks**2)
    conv_mod = nn.Conv2d(*shape[::-1], kernel_size=(ks, ks))
    conv_mod.weight.data = lin_mod.weight.data[..., None, None].reshape(conv_mod.weight.shape)
    conv_mod.bias.data = lin_mod.bias.data
    return conv_mod


class MyVGG11(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = Normalise()
        model = vgg11(pretrained=True)
        model.classifier[0] = (convert_lin2conv2d(model.classifier[0], ks=7))
        model.classifier[3] = (convert_lin2conv2d(model.classifier[3], ks=1))
        model.classifier[6] = (convert_lin2conv2d(model.classifier[6], ks=1))
        self.model = model

    def get_gcam_layer(self):
        return self.model.classifier

    def forward(self, x):
        x = self.norm(x)
        x = self.model.features(x)
        x = self.model.classifier(x)
        return x


class MyInception(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = Normalise()
        model = inception_v3(pretrained=True)
        model.fc = (convert_lin2conv2d(model.fc))
        self.model = model

    def get_gcam_layer(self):
        return self.model.fc

    def forward(self, x):
        x = self.norm(x)
        # N x 3 x 299 x 299
        x = self.model.Conv2d_1a_3x3(x)
        # N x 32 x 149 x 149
        x = self.model.Conv2d_2a_3x3(x)
        # N x 32 x 147 x 147
        x = self.model.Conv2d_2b_3x3(x)
        # N x 64 x 147 x 147
        x = self.model.maxpool1(x)
        # N x 64 x 73 x 73
        x = self.model.Conv2d_3b_1x1(x)
        # N x 80 x 73 x 73
        x = self.model.Conv2d_4a_3x3(x)
        # N x 192 x 71 x 71
        x = self.model.maxpool2(x)
        # N x 192 x 35 x 35
        x = self.model.Mixed_5b(x)
        # N x 256 x 35 x 35
        x = self.model.Mixed_5c(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_5d(x)
        # N x 288 x 35 x 35
        x = self.model.Mixed_6a(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6b(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6c(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6d(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_6e(x)
        # N x 768 x 17 x 17
        x = self.model.Mixed_7a(x)
        # N x 1280 x 8 x 8
        x = self.model.Mixed_7b(x)
        # N x 2048 x 8 x 8
        x = self.model.Mixed_7c(x)
        # N x 2048 x 8 x 8
        # Adaptive average pooling
        x = self.model.dropout(x)
        x = self.model.fc(x)
        # N x 1000 (num_classes)
        return x


class MyDenseNet121(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = Normalise()
        model = densenet121(pretrained=True)
        model.classifier = convert_lin2conv2d(model.classifier)
        self.model = model
        self.gcam_layer = "model.classifier"

    def get_gcam_layer(self):
        return self.model.classifier

    def forward(self, x):
        x = self.norm(x)
        features = self.model.features(x)
        out = F.relu(features, inplace=True)
        out = self.model.classifier(out)
        return out


class MyResNet34(nn.Module):

    def __init__(self):
        super().__init__()
        self.norm = Normalise()
        model = resnet34(pretrained=True)
        model.classifier = convert_lin2conv2d(model.fc)
        self.model = model

    def get_gcam_layer(self):
        return self.model.classifier

    def forward(self, x):
        x = self.norm(x)
        # See note [TorchScript super()]
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)

        x = self.model.classifier(x)

        return x

