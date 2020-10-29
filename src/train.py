import time
import os
import random
from utils import *
import torchvision
import matplotlib

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
        self.angle_template = torch.from_numpy(self.data.sample_images_from_index([0] + [x for x in range(27,1080,27)]))
        self.radius_template =  torch.from_numpy(self.data.sample_images_from_index([x for x in range(27)]))

        # from torchvision.utils import save_image
        # save_image(self.radius_template.view(-1, 1, 64, 64).type(torch.float32), 'ypos.png', padding=4, pad_value=1,
        #            nrow=27)
        self.angle_bins = np.array([-1] + [-1 + 2*x/40 for x in range(1,40)] +[1])
        self.radius_bins = np.array([-1] + [-1 + 2*x/27 for x in range(1,27)] + [1])


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

            g_optimizer.zero_grad()

            positive_angle_samples ,negative_angle_samples = self.get_sample_oracle_angle_pairs(c_cond)
            positive_radius_samples ,negative_radius_samples = self.get_sample_oracle_radius_pairs(c_cond)
            postive_pairs = torch.cat((positive_angle_samples,positive_radius_samples),dim=0).to(self.device)
            negative_pairs = torch.cat((negative_angle_samples, negative_radius_samples),dim=0).to(self.device)

            fake_x = model.decoder(z)
            total_images = torch.cat((fake_x,postive_pairs,negative_pairs),dim=0)
            latent_code, prob_fake , latent_similar = model.encoder(total_images)

            g_loss = adversarial_loss(prob_fake[:self.config['batch_size']], label_real)
            cont_loss = criterionQ_con(c_cond, latent_code[:self.config['batch_size']])
            similarity_loss_angle = similarity_loss(latent_similar[:self.config['batch_size']],latent_similar[30:60] ,latent_similar[90:120] )
            similarity_loss_radius = similarity_loss(latent_similar[:self.config['batch_size']],latent_similar[60:90] ,latent_similar[120:150] )
            G_loss = g_loss + cont_loss * self.config['lambda'] + 0.2*(similarity_loss_angle + similarity_loss_radius)
            G_loss.backward()

            g_optimizer.step()

            d_optimizer.zero_grad()
            latent_code, prob_real, _ = model.encoder(images)
            loss_real = adversarial_loss(prob_real, label_real)
            loss_real.backward()

            fake_x = model.decoder(z)
            latent_code_gen, prob_fake , _ = model.encoder(fake_x.detach())

            loss_fake = adversarial_loss(prob_fake, label_fake)
            loss_fake.backward()

            d_optimizer.step()

            D_loss = loss_real.item() + loss_fake.item()
            d_loss_summary = d_loss_summary + D_loss
            g_loss_summary = g_loss_summary + G_loss.item()
            info_loss_summary = info_loss_summary +  cont_loss.item()
        #
        logging.info("Epochs  %d / %d Time taken %d sec  G_Loss: %.5f, D_Loss %.5F Info_Loss %.5F Oracle_Loss %.5F" % (
            epoch, self.config['epochs'], time.time() - start_time,
            g_loss_summary / len(self.train_loader), d_loss_summary / len(self.train_loader), info_loss_summary / len(self.train_loader) ,oracle_loss_summary / len(self.train_loader)))
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


    def get_sample_oracle_radius_pairs(self , c_cond):
        # shape_factor = c_cond[:, 0]
        # c_cond_new = torch.rand(self.config['batch_size'], self.config['latent_dim'] -1, dtype=torch.float32,
        #                     device=self.device) * 2 - 1
        # c_cond_similar = torch.cat((shape_factor.view(-1,1), c_cond_new),dim=-1)
        latent = np.digitize(c_cond[:,0].cpu(),self.radius_bins)
        latent =  [x - 1 for x in latent.tolist()]
        replace_list = [random.choice(list(range(int(current_value))) + list(range(int(current_value) + 1, 27))) for current_value in latent]
        # differ_shape = torch.FloatTensor([random.uniform(bins[x],bins[x+1]) for x in  replace_list]).view(-1,1).to(self.device)
        # c_cond_differ = torch.cat((differ_shape, c_cond[:,1:]),dim=-1)
        negative_samples = torch.stack([self.radius_template[int(i)] for i in replace_list]).view(-1, 1, 64, 64)
        positive_samples = torch.stack([self.radius_template[int(i)] for i in latent]).view(-1, 1, 64, 64)
        return positive_samples ,negative_samples

    def get_sample_oracle_angle_pairs(self , c_cond):
        # size_factor = c_cond[:, 1
        # c_cond_new = torch.rand(self.config['batch_size'], self.config['latent_dim'] -1, dtype=torch.float32,
        #                     device=self.device) * 2 - 1
        # c_cond_similar = torch.cat((c_cond_new[:,:1], size_factor.view(-1,1), c_cond_new[:,1:]),dim=-1)
        latent = np.digitize(c_cond[:,1].cpu(),self.angle_bins)
        latent =  [x - 1 for x in latent.tolist()]

        replace_list = [random.choice(list(range(int(current_value))) + list(range(int(current_value) + 1, 40))) for current_value in latent]
        # differ_size = torch.FloatTensor([random.uniform(bins[x], bins[x + 1]) for x in replace_list]).view(-1, 1).to(self.device)
        # c_cond_differ = torch.cat((c_cond[:,:1] ,differ_size, c_cond[:, 2:]), dim=-1)
        negative_samples = torch.stack([self.angle_template[int(i)] for i in replace_list]).view(-1, 1, 64, 64)
        positive_samples = torch.stack([self.angle_template[int(i)] for i in latent]).view(-1, 1, 64, 64)
        return positive_samples ,negative_samples



