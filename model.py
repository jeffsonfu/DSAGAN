import torch
import torch.nn as nn
from torchvision.models import vgg19

class GAN(nn.Module):
    def __init__(self):
        super(GAN, self).__init__()
        self.conv00 = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0))
        self.conz00 = nn.Conv2d(1, 64, (1, 1), (1, 1), (0, 0))
        self.conv01 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))
        self.conz01 = nn.Conv2d(1, 64, (3, 3), (1, 1), (1, 1))
        self.conv02 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conz02 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))

        self.conv1 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conz1 = nn.Conv2d(64, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3))
        self.conz2 = nn.Conv2d(64, 64, (7, 7), (1, 1), (3, 3))
        self.conv3 = nn.Conv2d(128, 128, (1, 1), (1, 1), (0, 0))
        self.conv4 = nn.Conv2d(128, 64, (1, 1), (1, 1), (0, 0))
        self.conv5 = nn.Conv2d(64, 64, (1, 1), (1, 1), (0, 0))
        self.conv6 = nn.Conv2d(64, 1, (1, 1), (1, 1), (0, 0))

        self.up = nn.Upsample(scale_factor=2, mode='bicubic', align_corners=True)
        self.down = nn.AvgPool2d(2, 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.norm = nn.BatchNorm2d(128)
        self.nom = nn.BatchNorm2d(64)
        self.sig = nn.LeakyReLU()


    def forward(self, x, y):
        temx0 = self.conv00(x) + self.conv01(x) + self.conv02(x)
        temy0 = self.conz00(y) + self.conz01(y) + self.conz02(y)
        temx1 = self.conv1(temx0) * (1 + temx0)
        temy1 = self.conz1(temy0) * (1 + temy0)
        temx2 = self.conv2(temx1) * (1 + temx1 + temx0)
        temy2 = self.conz2(temy1) * (1 + temy1 + temy0)


        out = torch.cat((temx2, temy2), 1)
        out = self.norm(out)
        out = self.conv3(out)
        out = self.norm(out)
        out = self.conv4(out) * (1 + temx2 + temy2)
        out = self.nom(out)
        out = self.conv5(out)
        out = self.nom(out)
        out = self.conv6(out)
        #out3 = self.pool(temx0 + temx1 + temx2)
        #out3 = self.sig(out3)
        #out3 = torch.mul(out3, out)
        #out4 = self.pool(temy0 + temy1 + temy2)
        #out4 = self.sig(out4)
        #out4 = torch.mul(out4, out)
        #out = out + out1 + out2 + out3 + out4 + out5 + out6
        #out = self.sig(out)
        #out = self.conv5(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.net = nn.Sequential(
             nn.Conv2d(1, 64, kernel_size=1, padding=0),
             nn.LeakyReLU(),

             nn.Conv2d(64, 128, kernel_size=3, padding=1),
             nn.LeakyReLU(),

             nn.Conv2d(128, 256, kernel_size=3, padding=1),
             nn.LeakyReLU(),

             nn.Conv2d(256, 512, kernel_size=3, padding=1),
             nn.LeakyReLU(),

             nn.Conv2d(512, 512, kernel_size=3, padding=1),
             nn.BatchNorm2d(512),

             nn.AdaptiveMaxPool2d(1),
             nn.Conv2d(512, 1, kernel_size=1, padding=0)
        )


    def forward(self, x):
        batch_size = x.size(0)
        z = self.net(x).view(batch_size, 1)
        return torch.sigmoid(z)

