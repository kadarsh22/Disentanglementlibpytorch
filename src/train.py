import time
import os
import random
from utils import *
import numpy
from sklearn.model_selection import train_test_split
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
        start_time = time.time()
        d_loss_summary, g_loss_summary, info_loss_summary ,oracle_loss_summary = 0, 0, 0 , 0
        model.encoder.to(self.device)
        model.decoder.to(self.device)
        model.oracle.to(self.device)

        adversarial_loss = torch.nn.BCELoss()
        criterionQ_con = log_gaussian()
        similarity_loss = torch.nn.TripletMarginLoss()
        label_real = torch.full((self.config['batch_size'],), 1, dtype=torch.float32, device=self.device)
        label_fake = torch.full((self.config['batch_size'],), 0, dtype=torch.float32, device=self.device)

        for iter, images in enumerate(self.train_loader):
            images = images.type(torch.FloatTensor).to(self.device)
            z_noise = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                                 device=self.device) * 2 - 1
            c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                                device=self.device) * 2 - 1

            z = torch.cat((z_noise, c_cond), dim=1)

            positive_samples ,negative_samples = self.get_sample_pairs(c_cond)

            g_optimizer.zero_grad()

            fake_x = model.decoder(z)
            latent_code, prob_fake = model.encoder(fake_x)
            pos, neg, que = model.oracle(positive_samples ,negative_samples , fake_x)

            g_loss = adversarial_loss(prob_fake, label_real)
            cont_loss = criterionQ_con(c_cond, latent_code)
            oracle_loss = similarity_loss(que, pos, neg)

            G_loss = g_loss + cont_loss * 0.05 + oracle_loss
            G_loss.backward()

            g_optimizer.step()

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
            oracle_loss_summary = oracle_loss_summary + oracle_loss.item()
        #
        logging.info("Epochs  %d / %d Time taken %d sec  G_Loss: %.5f, D_Loss %.5F Info_Loss %.5F Oracle_Loss %.5F" % (
            epoch, self.config['epochs'], time.time() - start_time,
            g_loss_summary / len(self.train_loader), d_loss_summary / len(self.train_loader), info_loss_summary / len(self.train_loader) ,oracle_loss_summary / len(self.train_loader)))
        self.train_hist_gan['d_loss'].append(d_loss_summary/ len(self.train_loader))
        self.train_hist_gan['g_loss'].append(g_loss_summary/ len(self.train_loader))
        self.train_hist_gan['info_loss'].append(info_loss_summary/ len(self.train_loader))
        return model,self.train_hist_gan, (d_optimizer, g_optimizer)

    def train_classifier(self, model, optimizer, epoch):
        running_loss = 0.0
        model.to(self.device)
        criterion = torch.nn.TripletMarginLoss()
        for i, data in enumerate(self.train_loader, 0):
            positive ,negative ,query  = data
            positive = positive.cuda()
            negative = negative.cuda()
            query = query.cuda()

            optimizer.zero_grad()
            pos ,neg ,que  = model(positive ,negative ,query)
            loss = criterion(que, pos, neg)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        correct = 0
        total = 0
        for data  in self.test_loader:
            pdist = torch.nn.PairwiseDistance(p=2)
            positive, negative, query = data
            positive = positive.cuda()
            negative = negative.cuda()
            query = query.cuda()
            pos , neg ,query = model(positive ,negative ,query)
            dist_post = pdist(pos , query)
            dist_neg = pdist(neg,query)
            predicted = dist_post < dist_neg
            correct = correct + predicted.sum().item()

            total = total + query.size(0)
        print('Accuracy of the network on the 10000 test images: %d %% ' % ( 100 * correct / total))

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


    def get_sample_pairs(self , c_cond):
        bins = np.array([-1] + [-1 + 2*x/40 for x in range(1,40) ])
        latent = np.digitize(c_cond[:,3].cpu(),bins)
        latent =  [x - 1 for x in latent.tolist()]
        orientation_template = torch.from_numpy(self.data.sample_images_from_latent(self.data.latents_classes[[0] + [x for x in range(32 * 32 + 1, 40 * 32 * 32 + 1, 32 * 32)]]))
        replace_list = []
        for current_value in c_cond[:,3].tolist():
            replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) + 1, 40)))
            replace_list.append(replace_value)
        negative_samples = torch.stack([orientation_template[int(i)] for i in replace_list]).view(-1, 1, 64, 64)
        positive_samples = torch.stack([orientation_template[int(i)] for i in latent]).view(-1, 1, 64, 64)
        return positive_samples ,negative_samples
