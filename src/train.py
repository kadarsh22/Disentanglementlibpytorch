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
        categorical_loss = torch.nn.CrossEntropyLoss()
        label_real = torch.full((self.config['batch_size'],), 1, dtype=torch.float32, device=self.device)
        label_fake = torch.full((self.config['batch_size'],), 0, dtype=torch.float32, device=self.device)

        for iter, (images,labels) in enumerate(self.train_loader):
            cr_optimizer.zero_grad()

            z_noise_one = torch.rand(self.config['batch_size'], 62, dtype=torch.float32,
                                     device=self.device) * 2 - 1
            z_noise_two = torch.rand(self.config['batch_size'], 62, dtype=torch.float32,
                                     device=self.device) * 2 - 1

            fixed_idx = np.random.randint(low=0,high=10)
            target = torch.LongTensor([fixed_idx]*self.config['batch_size']).cuda()
            y_disc_ = torch.zeros((self.config['batch_size']*2,10)).cuda()
            y_disc_[:,fixed_idx] = 1
            inp_vec_one = torch.cat((z_noise_one, y_disc_[:self.config['batch_size']]), dim=1)
            inp_vec_two = torch.cat((z_noise_two, y_disc_[self.config['batch_size']:]), dim=1)
            inp_vec = torch.cat((inp_vec_one,inp_vec_two),dim=0)
            idx_fixed_data = model.decoder(inp_vec)
            cr_logits = model.cr_disc(idx_fixed_data[:self.config['batch_size']].detach(),
                                      idx_fixed_data[self.config['batch_size']:].detach())
            loss_cr_new = categorical_loss(cr_logits.view(-1,10), target)
            loss_cr_new.backward()
            cr_optimizer.step()


            images = images.type(torch.FloatTensor).to(self.device)
            z = torch.randn(self.config['batch_size'], 62, device=self.device)
            y_disc_ = torch.from_numpy(np.random.multinomial(1, 10* [float(1.0 / 10)], size=[self.config['batch_size']])).type(torch.FloatTensor).to(self.device)
            inp_vec = torch.cat((z,y_disc_),dim=1)


            d_optimizer.zero_grad()

            prob_real = model.encoder(images)[0]
            loss_D_real = adversarial_loss(prob_real, label_real)
            loss_D_real.backward()

            data_fake = model.decoder(inp_vec)

            prob_fake_D =  model.encoder(data_fake.detach())[0]
            loss_D_fake = adversarial_loss(prob_fake_D, label_fake)

            loss_D_fake.backward()
            loss_D = loss_D_real + loss_D_fake

            d_optimizer.step()

            g_optimizer.zero_grad()

            z = torch.randn(self.config['batch_size'], 62, device=self.device)
            y_disc_ = torch.from_numpy(np.random.multinomial(1,
                                                             10 * [float(1.0 / 10)], size=[self.config['batch_size']])).type(torch.FloatTensor).to(self.device)
            inp_vec = torch.cat((z, y_disc_), dim=1)
            _ , targets = torch.max(y_disc_,1)


            data_fake = model.decoder(inp_vec)
            prob_fake, disc_logits = model.encoder(data_fake)
            loss_G = adversarial_loss(prob_fake, label_real)
            loss_c_disc = categorical_loss(disc_logits.view(-1,10),targets)

            fixed_idx = np.random.randint(low=0, high=10)
            target = torch.LongTensor([fixed_idx] * self.config['batch_size']).cuda()
            z = torch.randn(self.config['batch_size'] * 2, 62, device=self.device)
            y_disc_ = torch.zeros((self.config['batch_size'] * 2, 10)).cuda()
            y_disc_[:, fixed_idx] = 1
            inp_vec = torch.cat((z, y_disc_), dim=1)
            idx_fixed_data = model.decoder(inp_vec)
            cr_logits = model.cr_disc(idx_fixed_data[:self.config['batch_size']],
                                      idx_fixed_data[self.config['batch_size']:])
            loss_cr =  categorical_loss(cr_logits.view(-1,10), target)

            loss_info = loss_G + loss_c_disc + 2*loss_cr

            loss_info.backward()
            g_optimizer.step()



            d_loss_summary = d_loss_summary + loss_D.item()
            g_loss_summary = g_loss_summary + loss_G.item()
            info_loss_summary = info_loss_summary +  loss_info.item()
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
        z = torch.rand(self.config['batch_size'], 62, device=self.device)
        y_disc_ = torch.from_numpy(np.random.multinomial(1, 10 * [float(1.0 / 10)], size=self.config['batch_size'])).type(torch.FloatTensor).cuda()
        return z , y_disc_


    def _get_training_data(self):
        if self.config['dataset'] != 'fashion_mnist':
            images = self.data.images
            train_loader = torch.utils.data.DataLoader(images, batch_size=self.config['batch_size'], shuffle=True)
            return train_loader
        else:
            train_loader = torch.utils.data.DataLoader(self.data, batch_size=self.config['batch_size'], shuffle=True,drop_last=True)
            return train_loader