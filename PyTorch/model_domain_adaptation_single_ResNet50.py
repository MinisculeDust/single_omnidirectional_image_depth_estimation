import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        # Store context for backprop
        ctx.alpha = alpha

        # Forward pass is a no-op
        return x.view_as(x) # view_as return the same scale tensor

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass is just to -alpha the gradient
        output = grad_output.neg() * ctx.alpha # .neg() Returns a new tensor with the negative of the elements of input

        # Must return same number as inputs to forward()
        return output, None

class UpSample(nn.Sequential):
    def __init__(self, skip_input, output_features):
        super(UpSample, self).__init__()
        self.convA = nn.Conv2d(skip_input, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluA = nn.LeakyReLU(0.2)
        self.convB = nn.Conv2d(output_features, output_features, kernel_size=3, stride=1, padding=1)
        self.leakyreluB = nn.LeakyReLU(0.2)

    def forward(self, x, concat_with):
        up_x = F.interpolate(x, size=[concat_with.size(2), concat_with.size(3)], mode='bilinear', align_corners=True)
        return self.leakyreluB( self.convB( self.leakyreluA(self.convA( torch.cat([up_x, concat_with], dim=1) ) ) )  )



class Decoder(nn.Module):
    def __init__(self, num_features=2048, decoder_width = 0.5):
        super(Decoder, self).__init__()

        features = int(num_features * decoder_width) # features: 1024

        self.conv2 = nn.Conv2d(num_features, features, kernel_size=1, stride=1, padding=1)

        # skip_input == 2048, output_features=512
        self.up1 = UpSample(skip_input=features//1 + 1024, output_features=features//2)
        # skip_input == 1024, output_features=256
        self.up2 = UpSample(skip_input=features//2 + 512, output_features=features//4)
        # skip_input == 320, output_features=128
        self.up3 = UpSample(skip_input=features//4 + 64, output_features=features//8)
        # skip_input == 192, output_features=64
        self.up4 = UpSample(skip_input=features//8 + 64, output_features=features//16)

        # features//16 = 64
        self.conv3 = nn.Conv2d(features//16, 1, kernel_size=3, stride=1, padding=1)

    def forward(self, features):
        x_block0, x_block1, x_block2, x_block3, x_block4 = features[3], features[4], features[6], features[7], features[8]
        x_d0 = self.conv2(x_block4) # torch.Size([bs, 2048, 8, 16]) ==> torch.Size([bs, 1024, 10, 18])
        x_d1 = self.up1(x_d0, x_block3) # torch.Size([bs, 1024, 10, 18]) ==> torch.Size([bs, 512, 16, 32])
        x_d2 = self.up2(x_d1, x_block2) # torch.Size([bs, 256, 32, 64])
        x_d3 = self.up3(x_d2, x_block1) # torch.Size([bs, 128, 64, 128])
        x_d4 = self.up4(x_d3, x_block0) # torch.Size([bs, 64, 128, 256])
        return self.conv3(x_d4) # torch.Size([bs, 1, 128, 256])


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        import torchvision.models as models
        self.original_model = models.resnet50(pretrained=False)

    def forward(self, x, grl_lambda):

        features = [x]
        for k, v in self.original_model._modules.items():
            if k == 'avgpool' or k == 'fc': # only gain the value before the layer 'avgpool' and 'fc'
                break
            features.append( v(features[-1]) ) # compute and record output of each layer

        domain_features = features[-1].view(-1, 2048 * 8 * 16) # features[-1]: torch.Size([bs, 2048, 8, 16])
        reverse_features = GradientReversalFn.apply(domain_features, grl_lambda)

        return features, reverse_features # output size: torch.Size([5, 2208, 8, 16])

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(2048 * 128, 128)
        self.fc2 = nn.Linear(128, 2)
        # self.sigmoid = nn.Sigmoid()
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, x):

        out = self.fc1(x)
        out = F.relu(out)
        out = self.fc2(out)
        # out = self.sigmoid(out)
        out = self.logsoftmax(out)
        return out


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()
        self.classifier = Classifier()

    def forward(self, x, grl_lambda=1.0):

        imbedded_features, reverse_features = self.encoder(x, grl_lambda)
        output = self.decoder( imbedded_features )
        pre_domain = self.classifier( reverse_features )
        return output, pre_domain

