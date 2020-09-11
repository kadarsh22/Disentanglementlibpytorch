import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.nc =1
        self.cnn1_en = nn.Conv2d(1, 32, 4, 2, 1)
        self.cnn2_en = nn.Conv2d(32, 32, 4, 2, 1)
        self.cnn3_en = nn.Conv2d(32, 32, 4, 2, 1)
        self.cnn4_en = nn.Conv2d(32, 32, 4, 2, 1)
        self.linear1_en = nn.Linear(32 * 4 * 4, 256)
        self.linear2_en = nn.Linear(256, 256)
        self.z_mean = nn.Linear(256, 5)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = x.type(torch.cuda.FloatTensor)
        x = x.view(-1, self.nc, 64, 64)
        out = self.relu(self.cnn1_en(x))
        out = self.relu(self.cnn2_en(out))
        features = self.relu(self.cnn3_en(out))
        out = self.relu(self.cnn4_en(features)).view(-1, 32 * 4 * 4)
        out = self.relu(self.linear1_en(out))
        out = self.relu(self.linear2_en(out))
        representation = self.z_mean(out)
        return representation , features

class InfoMax(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.global_discriminator = GlobalDiscriminator().cuda()
        self.local_discriminator = LocalDiscriminator().cuda()
        self.prior = PriorDiscriminator().cuda()


class GlobalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(32, 64, kernel_size=3)
        self.c1 = nn.Conv2d(64, 32, kernel_size=3)
        self.l0 = nn.Linear(512+5, 512)
        self.l1 = nn.Linear(512, 512)
        self.l2 = nn.Linear(512, 1)

    def forward(self, y, M):
        h = F.relu(self.c0(M))
        h = self.c1(h)
        h = h.view(y.shape[0], -1)
        h = torch.cat((y, h), dim=1)
        h = F.relu(self.l0(h))
        h = F.relu(self.l1(h))
        return self.l2(h)


class LocalDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(37, 512, kernel_size=1)
        self.c1 = nn.Conv2d(512, 512, kernel_size=1)
        self.c2 = nn.Conv2d(512, 1, kernel_size=1)

    def forward(self, x):
        h = F.relu(self.c0(x))
        h = F.relu(self.c1(h))
        return self.c2(h)


class PriorDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.l0 = nn.Linear(5, 1000)
        self.l1 = nn.Linear(1000, 200)
        self.l2 = nn.Linear(200, 1)

    def forward(self, x):
        h = F.relu(self.l0(x))
        h = F.relu(self.l1(h))
        return torch.sigmoid(self.l2(h))
