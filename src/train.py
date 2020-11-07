import time
import os
import random
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
        cr_optimizer = optimizer[2]
        start_time = time.time()
        d_loss_summary, g_loss_summary, info_loss_summary ,oracle_loss_summary = 0, 0, 0 , 0
        model.encoder.to(self.device)
        model.decoder.to(self.device)
        model.cr_disc.to(self.device)

        adversarial_loss = torch.nn.BCELoss()
        criterionQ_con = log_gaussian()
        categorical_loss = torch.nn.CrossEntropyLoss()
        label_real = torch.full((self.config['batch_size'],), 1, dtype=torch.float32, device=self.device)
        label_fake = torch.full((self.config['batch_size'],), 0, dtype=torch.float32, device=self.device)

        for iter, (images,labels) in enumerate(self.train_loader):
            images = images.type(torch.FloatTensor).to(self.device)

            z , idx = self._sample()

            g_optimizer.zero_grad()

            fake_x = model.decoder(z)
            prob_fake, disc_logits, latent_code, c_cont_var = model.encoder(fake_x)

                # Calculate loss for discrete latent code
            target = torch.LongTensor(idx).to(self.device)
            loss_c_disc = 0
            for j in range(1):
                loss_c_disc += categorical_loss(disc_logits[:, j, :], target[j, :])

            g_loss = adversarial_loss(prob_fake, label_real)
            cont_loss = criterionQ_con(z[:, self.config['noise_dim']+self.config['discrete_dim']:], latent_code)

            G_loss = g_loss + cont_loss.sum() * self.config['lambda'] + loss_c_disc
            G_loss.backward()

            g_optimizer.step()

            d_optimizer.zero_grad()
            prob_real ,_ ,_ ,_ = model.encoder(images)
            loss_real = adversarial_loss(prob_real, label_real)
            loss_real.backward()

            fake_x = model.decoder(z)
            prob_fake ,_ ,_,_ = model.encoder(fake_x.detach())

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
        return model,self.train_hist_gan, (d_optimizer, g_optimizer,cr_optimizer)

    @staticmethod
    def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(seed)
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    def _sample(self):
        z = torch.randn(self.config['batch_size'], self.config['noise_dim'], device=self.device)
        idx = np.zeros((self.config['discrete_dim'], self.config['batch_size']))
        c_disc = torch.zeros(self.config['batch_size'], 1, self.config['discrete_dim'], device=self.device)
        for i in range(1):
            idx[i] = np.random.randint(self.config['discrete_dim'], size=self.config['batch_size'])
            c_disc[torch.arange(0, self.config['batch_size']), i, idx[i]] = 1.0

        c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'], device=self.device) * 2 - 1

        for i in range(1):
            z = torch.cat((z, c_disc[:, i, :].squeeze()), dim=1)

        z = torch.cat((z, c_cond), dim=1)
        return z ,idx


    def _get_training_data(self):
        if self.config['dataset'] != 'fashion_mnist':
            images = self.data.images
            train_loader = torch.utils.data.DataLoader(images, batch_size=self.config['batch_size'], shuffle=True)
            return train_loader
        else:
            train_loader = torch.utils.data.DataLoader(self.data, batch_size=self.config['batch_size'], shuffle=True)
            return train_loader