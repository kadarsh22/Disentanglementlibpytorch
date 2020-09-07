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
        self.loss = DeepInfoMaxLoss()



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

class DeepInfoMaxLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=1.0, gamma=0.1):
        super().__init__()
        self.global_d = GlobalDiscriminator().cuda()
        self.local_d = LocalDiscriminator().cuda()
        self.prior_d = PriorDiscriminator().cuda()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, y, M, M_prime):

        # see appendix 1A of https://arxiv.org/pdf/1808.06670.pdf

        y_exp = y.unsqueeze(-1).unsqueeze(-1)
        y_exp = y_exp.expand(-1, -1, 8, 8)

        y_M = torch.cat((M, y_exp), dim=1)
        y_M_prime = torch.cat((M_prime, y_exp), dim=1)

        Ej = -F.softplus(-self.local_d(y_M)).mean()
        Em = F.softplus(self.local_d(y_M_prime)).mean()
        LOCAL = (Em - Ej) * self.beta

        Ej = -F.softplus(-self.global_d(y, M)).mean()
        Em = F.softplus(self.global_d(y, M_prime)).mean()
        GLOBAL = (Em - Ej) * self.alpha

        prior = torch.rand_like(y)

        term_a = torch.log(self.prior_d(prior)+1e-9).mean()
        term_b = torch.log(1.0 - self.prior_d(y.detach())).mean()
        PRIOR = - (term_a + term_b) * self.gamma


        return LOCAL + GLOBAL + PRIOR , term_b
