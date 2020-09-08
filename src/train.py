import time
import os
import random
from utils import *
import torch.nn.functional as F

# from sklearn.preprocessing import train_test_split

log = logging.getLogger(__name__)


class Trainer(object):

    def __init__(self, dsprites, config):
        super(Trainer, self).__init__()
        self.data = dsprites
        self.config = config
        self.device = torch.device('cuda:' + str(config['device_id']))
        self.train_loader = self._get_training_data()
        # self.train_loader , self.test_loader = self._get_classifier_training_data()
        self.train_hist_vae = {'loss': [], 'bce_loss': [], 'kld_loss': []}
        self.train_hist_gan = {'d_loss': [], 'g_loss': [], 'info_loss': [],'cr_loss':[]}

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

    def train_deepinfomax(self, model ,optimizer, epoch):
        """
            Deep Infomax training
        :param model: Encoder Model
        :param optimizer: tuple consisting of loss optimizer
        :param epoch: epoch
        :return:
        """

        train_loss = 0
        start_time = time.time()
        adversarial_loss = torch.nn.BCELoss()
        label_real = torch.full((self.config['batch_size'],), 1, dtype=torch.float32, device=self.device)
        label_fake = torch.full((self.config['batch_size'],), 0, dtype=torch.float32, device=self.device)

        enocder_optim, global_loss_optim , local_loss_optim , prior_loss_optim = optimizer
        for images in self.train_loader:
            images = images.to(self.device)
            enocder_optim.zero_grad()
            local_loss_optim.zero_grad()
            y, M = model.encoder(images)
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)

            y_exp = y.unsqueeze(-1).unsqueeze(-1)
            y_exp = y_exp.expand(-1, -1, 8, 8)

            y_M = torch.cat((M, y_exp), dim=1)
            y_M_prime = torch.cat((M_prime, y_exp), dim=1)
            Ej = -F.softplus(-model.local_discriminator(y_M)).mean()
            Em = F.softplus(model.local_discriminator(y_M_prime)).mean()
            local_loss = (Em - Ej)
            local_loss.backward()
            local_loss_optim.step()
            enocder_optim.step()

            enocder_optim.zero_grad()
            global_loss_optim.zero_grad()

            y, M = model.encoder(images)
            M_prime = torch.cat((M[1:], M[0].unsqueeze(0)), dim=0)
            Ej = -F.softplus(-model.global_discriminator(y, M)).mean()
            Em = F.softplus(model.global_discriminator(y, M_prime)).mean()
            global_loss = (Em - Ej) * 0.5
            global_loss.backward()
            global_loss_optim.step()
            enocder_optim.step()

            prior_loss_optim.zero_grad()

            prior = torch.rand_like(y)
            prob_real = model.prior(prior)
            loss_real = adversarial_loss(prob_real, label_real)
            loss_real.backward()

            prob_fake = model.prior(y.detach())
            loss_fake = adversarial_loss(prob_fake , label_fake)
            loss_fake.backward()
            prior_loss_optim.step()

            enocder_optim.zero_grad()
            y, _ = model.encoder(images)
            label = model.prior(y)
            loss = adversarial_loss(label, label_real.view(-1,1))
            loss.backward()
            enocder_optim.step()


        logging.info("Epochs  %d / %d Time taken %d sec Loss : %.3f " % (
            epoch, self.config['epochs'], time.time() - start_time, train_loss / len(self.train_loader)))
        return model, (enocder_optim, global_loss_optim ,local_loss_optim ,prior_loss_optim)


    def train_gan(self, model, optimizer, epoch):
        d_optimizer = optimizer[0]
        g_optimizer = optimizer[1]
        cr_optimizer = optimizer[2]
        start_time = time.time()
        d_loss_summary, g_loss_summary, cr_loss, info_loss_summary = 0, 0, 0, 0
        adversarial_loss = torch.nn.BCELoss()
        criterionQ_con = log_gaussian()
        categorical_loss = torch.nn.CrossEntropyLoss()

        label_real = torch.full((self.config['batch_size'],), 1, dtype=torch.float32, device=self.device)
        label_fake = torch.full((self.config['batch_size'],), 0, dtype=torch.float32, device=self.device)


        for iter, images in enumerate(self.train_loader):

            # # CR Loss
            cr_optimizer.zero_grad()
            idx_fixed_z, fixed_idx = self._get_fixed_idx(self.config['cr_gap'])
            idx_fixed_data = model.decoder(idx_fixed_z)
            cr_logits = model.cr_disc(idx_fixed_data[:self.config['batch_size']].detach(),
                                      idx_fixed_data[self.config['batch_size']:].detach())
            loss_cr_new = categorical_loss(cr_logits, fixed_idx)
            loss_cr_new.backward()
            cr_optimizer.step()


            images = images.type(torch.FloatTensor).to(self.device)
            z_noise = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                                 device=self.device) * 2 - 1
            c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                                device=self.device) * 2 - 1

            z = torch.cat((z_noise, c_cond), dim=1)

            ## decoder optimizatiom

            g_optimizer.zero_grad()

            fake_x = model.decoder(z)
            latent_code, prob_fake = model.encoder(fake_x)

            g_loss = adversarial_loss(prob_fake, label_real)
            cont_loss = criterionQ_con(c_cond, latent_code)

            idx_fixed_z, fixed_idx = self._get_fixed_idx(self.config['cr_gap'])
            idx_fixed_data = model.decoder(idx_fixed_z)
            cr_logits = model.cr_disc(idx_fixed_data[:self.config['batch_size']],
                                      idx_fixed_data[self.config['batch_size']:])
            loss_cr =  categorical_loss(cr_logits, fixed_idx)


            G_loss = g_loss + (cont_loss * 0.05) + (loss_cr)
            G_loss.backward()

            g_optimizer.step()

            ## encoder optimization
            d_optimizer.zero_grad()
            latent_code, prob_real = model.encoder(images)
            loss_real = adversarial_loss(prob_real, label_real)
            loss_real.backward()

            fake_x = model.decoder(z)
            latent_code_gen, prob_fake = model.encoder.forward_no_spectral(fake_x)

            loss_fake = adversarial_loss(prob_fake, label_fake)
            q_loss = criterionQ_con(c_cond, latent_code_gen)
            loss = loss_fake + 0.05 * q_loss
            loss.backward()

            D_loss = loss_real + loss
            d_optimizer.step()


            d_loss_summary = d_loss_summary + D_loss.item()
            g_loss_summary = g_loss_summary + G_loss.item()
            info_loss_summary = info_loss_summary + q_loss.item() + cont_loss.item()
            cr_loss = cr_loss + loss_cr_new.item()
        #
        logging.info("Epochs  %d / %d Time taken %d sec  D_Loss: %.5f, G_Loss %.5F , CR Loss %.5F" % (
            epoch, self.config['epochs'], time.time() - start_time,
            g_loss / len(self.train_loader), d_loss_summary / len(self.train_loader),
            cr_loss / len(self.train_loader)))
        self.train_hist_gan['d_loss'].append(d_loss_summary / len(self.train_loader))
        self.train_hist_gan['g_loss'].append(g_loss_summary / len(self.train_loader))
        self.train_hist_gan['info_loss'].append(info_loss_summary / len(self.train_loader))
        self.train_hist_gan['cr_loss'].append(cr_loss / len(self.train_loader))
        return model, self.train_hist_gan, (d_optimizer, g_optimizer, cr_optimizer)

    def _get_fixed_idx(self, gap):
        a, b = -1 + gap / 2, 1 + gap / 2
        idx_fixed = torch.from_numpy(np.array([np.random.randint(0, self.config['latent_dim'])
                                               for i in range(self.config['batch_size'])])).to(self.device)
        c_cond_one = torch.rand(self.config['batch_size'], 1, dtype=torch.float32,
                                device=self.device) * 2 - 1
        latent_one = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                                device=self.device) * (b - a) + a
        latent_two = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                                device=self.device) * (b - a) + a
        # latent_one[latent_one - latent_two > 0] = latent_one[latent_one - latent_two > 0] + gap / 2
        # latent_two[latent_one - latent_two > 0] = latent_two[latent_one - latent_two > 0] - gap / 2
        # latent_one[latent_two - latent_one > 0] = latent_one[latent_two - latent_one > 0] - gap / 2
        # latent_two[latent_two - latent_one > 0] = latent_two[latent_two - latent_one > 0] + gap / 2
        for i in range(idx_fixed.shape[0]):
            latent_one[i][idx_fixed[i]] = c_cond_one[i]
            latent_two[i][idx_fixed[i]] = c_cond_one[i]

        z_noise_one = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                                 device=self.device) * 2 - 1
        z_noise_two = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                                 device=self.device) * 2 - 1

        latent_pair_one = torch.cat((z_noise_one, latent_one), dim=1)
        latent_pair_two = torch.cat((z_noise_two, latent_two), dim=1)

        z = torch.cat((latent_pair_one, latent_pair_two), dim=0).to(self.device)
        return z, idx_fixed

    def train_classifier(self, model, optimizer, epoch):
        running_loss = 0.0
        criterion = torch.nn.CrossEntropyLoss()
        for i, data in enumerate(self.train_loader, 0):

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % 2000 == 1999:  # print every 2000 mini-batches
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 2000))
                running_loss = 0.0

        correct = 0
        total = 0
        with torch.no_grad():
            for data in self.test_loader:
                images, labels = data
                outputs = model(images.cuda())
                _, predicted = torch.max(outputs.data.cpu(), 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print('Accuracy of the network on the 10000 test images: %d %%' % (
                100 * correct / total))
        return model, optimizer, epoch

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

    def _get_classifier_training_data(self):
        images = self.data.images
        labels = torch.from_numpy(self.data.latents_classes[:, 3]).type(torch.LongTensor)
        X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
        train_dataset = NewDataset(torch.from_numpy(X_train), y_train)
        test_dataset = NewDataset(torch.from_numpy(X_test), y_test)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['batch_size'], shuffle=True)
        return train_loader, test_loader

    def _sample(self):
        z_noise = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                             device=self.device) * 2 - 1
        c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                            device=self.device) * 2 - 1

        z = torch.cat((z_noise, c_cond), dim=1)
        return z
