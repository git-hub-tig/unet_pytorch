import math
import torch
import torchvision.models as tvm


class UNet(torch.nn.Module):
    def __init__(self, decoder_activation_fn):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder(activation_fn=decoder_activation_fn)
    def forward(self, image):
        return self.decoder(self.encoder(image))


class Encoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = torch.nn.Sequential(*list(tvm.resnet18(pretrained=True).children())[:8])
        self.layers = [2,3,5,6, 7]
    
    def forward(self, image):
        features = [image]
        for _, layer in enumerate(self.features.children()): features.append(layer(features[-1]))
        return [features[layer] for layer in self.layers]


class Decoder(torch.nn.Module):
    def __init__(self, encoder_channels=[64, 64, 64, 128, 256], n_channels=15, activation_fn: str = "gelu"):
        super().__init__()
        self.up4 = Upsample(
            in_ch=encoder_channels[3]+encoder_channels[4], mid_ch=encoder_channels[3], out_ch=encoder_channels[2], activation=activation_fn
        )
        self.up3 = Upsample(
            in_ch=2*encoder_channels[2], mid_ch=encoder_channels[2]//2, out_ch=encoder_channels[1], activation=activation_fn
        )
        self.up2 = Upsample(
            in_ch=2*encoder_channels[1], mid_ch=encoder_channels[1]//2, out_ch=encoder_channels[0], activation=activation_fn
        )
        self.up1 = Upsample(
            in_ch=2*encoder_channels[0], mid_ch=encoder_channels[0]//2, out_ch=64, activation=activation_fn
        )
        self.up0 = Upsample(
            in_ch=64, mid_ch=32, out_ch=n_channels, activation=activation_fn
        )
    
    def forward(self, features):
        x1, x2, x3, x4, x5 = features
        
        hx5 = up2(x5)
        hx4 = self.up4(torch.cat([x4, hx5], 1))
        hx4up = up2(hx4)
        hx3 = self.up3(torch.cat([x3, hx4up], 1))
        hx3up = up2(hx3)
        hx2 = self.up2(torch.cat([x2, hx3up], 1))
        hx1 = self.up1(torch.cat([x1, hx2], 1))
        
        hx0 = self.up0(up2(hx1))
        return torch.sigmoid(hx0)


class Upsample(torch.nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch, activation: str = "gelu"):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_ch, mid_ch, kernel_size=3, stride=1, padding=1)
        self.activation_fn = ACTIVATIONS[activation]
        self.bn1 = torch.nn.BatchNorm2d(mid_ch)
        self.conv2 = torch.nn.Conv2d(mid_ch, out_ch, kernel_size=3, stride=1, padding=1)
    def forward(self, feat):
        return self.conv2(self.activation_fn(self.bn1(self.conv1(feat))))


def up2(x):
    return torch.nn.functional.interpolate(x, scale_factor=(2,2), mode="bilinear", align_corners=False)


def gelu_new(x):
    """
    Implementation of the GELU activation function currently in Google BERT repo (identical to OpenAI GPT). Also see
    the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """
    return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * torch.pow(x, 3.0))))


def gelu_fast(x):
    return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))


def quick_gelu(x):
    return x * torch.sigmoid(1.702 * x)

ACTIVATIONS = {
    "relu": torch.nn.functional.relu,
    "gelu": torch.nn.functional.gelu,
    "quick_relu": quick_gelu,
    "gelu_fast":gelu_fast,
    "gelu_new": gelu_new,
} 