import time
import os
import random
import numpy as np
from utils import *

log = logging.getLogger(__name__)

class Trainer(object):

    def __init__(self, dsprites, config):
        super(Trainer, self).__init__()
        self.data = dsprites
        self.config = config
        self.device = torch.device('cuda:' + str(config['device_id']))
        self.train_loader = self._get_training_data()
        self.train_hist_vae = {'loss': [], 'bce_loss': [], 'kld_loss': []}
        self.train_hist_gan = {'d_loss': [], 'g_loss': [], 'info_loss': []}
        self.shape = [x for x in range(3)]
        self.size = [x for x in range(6)]
        self.orient = [x for x in range(40)]
        self.xpos = [x for x in range(32)]
        self.ypos = [x for x in range(32)]

    def train_vae(self, model, optimizer, epoch):
        start_time = time.time()
        bce_loss, kld_loss, total_loss = 0, 0, 0
        for images in self.train_loader:
            images = images.to(self.device)
            optimizer.zero_grad()
            loss, out = model(images)
            loss[0].backward()
            optimizer.step()
            bce_loss = bce_loss + loss[2].item()
            kld_loss = kld_loss + loss[1].item()
            total_loss = total_loss + loss[0].item()
        logging.info("Epochs  %d / %d Time taken %d sec Loss : %.3f BCELoss: %.3f, KLDLoss %.3F" % (
            epoch, self.config['epochs'], time.time() - start_time, total_loss / len(self.train_loader),
            bce_loss / len(self.train_loader), kld_loss / len(self.train_loader)))
        self.train_hist_vae['loss'].append(total_loss / self.config['batch_size'])
        self.train_hist_vae['bce_loss'].append(bce_loss / self.config['batch_size'])
        self.train_hist_vae['kld_loss'].append(kld_loss / self.config['batch_size'])
        return model, self.train_hist_vae, (optimizer,)

    def train_gan(self, model, optimizer, epoch):
        d_optimizer = optimizer[0]
        g_optimizer = optimizer[1]
        start_time = time.time()
        d_loss_summary, g_loss_summary, info_loss_summary ,oracle_loss_summary = 0, 0, 0 , 0
        model.encoder.to(self.device)
        model.decoder.to(self.device)
        # model.mode_counter.to(self.device)

        adversarial_loss = torch.nn.BCELoss()
        categorical_loss = torch.nn.CrossEntropyLoss()
        label_real = torch.full((self.config['batch_size'],), 1, dtype=torch.float32, device=self.device)
        label_fake = torch.full((self.config['batch_size'],), 0, dtype=torch.float32, device=self.device)

        for iter, images in enumerate(self.train_loader):
            images = images.type(torch.FloatTensor).to(self.device)
            z_noise = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                                 device=self.device) * 2 - 1
            c_cond , shape_mu ,size_mu,orient_mu ,xpos_mu ,ypos_mu = self.sample()

            z = torch.cat((z_noise, c_cond), dim=1)

            g_optimizer.zero_grad()

            fake_x = model.decoder(z)
            prob_fake , latent_code = model.encoder(fake_x)
            c_disc_shape, c_disc_size, c_disc_orient, c_disc_xpos, c_disc_ypos = latent_code

            g_loss = adversarial_loss(prob_fake, label_real)

            shape_loss = categorical_loss(c_disc_shape.view(-1,3), shape_mu.view(-1))
            size_loss = categorical_loss(c_disc_size.view(-1, 6), size_mu.view(-1))
            orient_loss = categorical_loss(c_disc_orient.view(-1, 40), orient_mu.view(-1))
            xpos_loss = categorical_loss(c_disc_xpos.view(-1, 32), xpos_mu.view(-1))
            ypos_loss = categorical_loss(c_disc_ypos.view(-1, 32), ypos_mu.view(-1))

            G_loss = g_loss +  self.config['lambda']
            G_loss.backward()

            g_optimizer.step()

            d_optimizer.zero_grad()
            latent_code, prob_real = model.encoder(images)
            loss_real = adversarial_loss(prob_real, label_real)
            loss_real.backward()

            fake_x = model.decoder(z)
            latent_code_gen, prob_fake = model.encoder(fake_x.detach())

            loss_fake = adversarial_loss(prob_fake, label_fake)
            loss_fake.backward()

            d_optimizer.step()

            D_loss = loss_real.item() + loss_fake.item()
            d_loss_summary = d_loss_summary + D_loss
            g_loss_summary = g_loss_summary + G_loss.item()
            info_loss_summary = info_loss_summary +  cont_loss.item()
        #
        logging.info("Epochs  %d / %d Time taken %d sec  G_Loss: %.5f, D_Loss %.5F Info_Loss %.5F F" % (
            epoch, self.config['epochs'], time.time() - start_time,
            g_loss_summary / len(self.train_loader), d_loss_summary / len(self.train_loader), info_loss_summary / len(self.train_loader)))
        self.train_hist_gan['d_loss'].append(d_loss_summary/ len(self.train_loader))
        self.train_hist_gan['g_loss'].append(g_loss_summary/ len(self.train_loader))
        self.train_hist_gan['info_loss'].append(info_loss_summary/ len(self.train_loader))
        return model,self.train_hist_gan, (d_optimizer, g_optimizer)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _get_training_data(self):
        images = self.data.images
        train_loader = torch.utils.data.DataLoader(images, batch_size=self.config['batch_size'], shuffle=True)
        return train_loader

    def sample(self):
        shape_mu = torch.LongTensor(np.random.choice(self.shape,size=64)).view(-1,1).to(self.device)
        size_mu = torch.LongTensor(np.random.choice(self.size, size=64)).view(-1,1).to(self.device)
        orient_mu = torch.LongTensor(np.random.choice(self.orient, size=64)).view(-1,1).to(self.device)
        xpos_mu = torch.LongTensor(np.random.choice(self.xpos, size=64)).view(-1,1).to(self.device)
        ypos_mu = torch.LongTensor(np.random.choice(self.ypos, size=64)).view(-1,1).to(self.device)
        shape_onehot =torch.nn.functional.one_hot(shape_mu.to(torch.int64),num_classes=3)
        size_onehot = torch.nn.functional.one_hot(size_mu.to(torch.int64), num_classes=6)
        orient_onehot = torch.nn.functional.one_hot(orient_mu.to(torch.int64), num_classes=40)
        xpos_onehot = torch.nn.functional.one_hot(xpos_mu.to(torch.int64), num_classes=32)
        ypos_onehot = torch.nn.functional.one_hot(ypos_mu.to(torch.int64), num_classes=32)
        c_cond = torch.cat((shape_onehot,size_onehot,orient_onehot,xpos_onehot,ypos_onehot),dim=-1).squeeze()
        return c_cond.to(self.device) ,shape_mu ,size_mu,orient_mu ,xpos_mu ,ypos_mu

