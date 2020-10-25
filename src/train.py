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
        self.shape_template = torch.from_numpy(self.data.sample_images_from_latent(self.data.latents_classes[[0] + [x for x in range(6 * 40 * 32 * 32 + 1, 3 * 6 * 40 * 32 * 32 + 1, 6 * 40 * 32 * 32)]]))
        self.size_template =  torch.from_numpy(self.data.sample_images_from_latent(self.data.latents_classes[[0] + [x for x in range(40 * 32 * 32 + 1, 6 * 40 * 32 * 32 + 1, 40 * 32 * 32)]]))
        self.orientation_template = torch.from_numpy(self.data.sample_images_from_latent(self.data.latents_classes[[0] + [x for x in range(32 * 32 + 1, 40 * 32 * 32 + 1, 32 * 32)]]))
        self.xpos_template = torch.from_numpy(self.data.sample_images_from_latent(self.data.latents_classes[[0] + [x for x in range(32, 32 * 32, 32)]]))
        self.ypos_template = torch.from_numpy(self.data.sample_images_from_latent(self.data.latents_classes[[x for x in range(32)]]))

        self.shape_bins = np.array([-1] + [-1 + 2*x/3 for x in range(1,3)] +[1])
        self.size_bins = np.array([-1] + [-1 + 2*x/6 for x in range(1,6)] + [1])
        self.orientation_bins = np.array([-1] + [-1 + 2*x/40 for x in range(1,40)] + [1])
        self.xpos_bins =  np.array([-1] + [-1 + 2*x/32 for x in range(1,32) ] + [1])
        self.ypos_bins =  np.array([-1] + [-1 + 2*x/32 for x in range(1,32)] + [1])
        # from torchvision.utils import save_image
        # save_image(self.shape_template.view(-1, 1, 64, 64).type(torch.float32), 'ypos.png', padding=4, pad_value=1,
        #            nrow=32)

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

            positive_shape_samples ,negative_shape_samples = self.get_sample_oracle_shape_pairs(c_cond)
            positive_size_samples ,negative_size_samples = self.get_sample_oracle_size_pairs(c_cond)
            positive_orient_samples ,negative_orient_samples = self.get_sample_oracle_orient_pairs(c_cond)
            positive_xpos_samples ,negative_xpos_samples= self.get_sample_oracle_xpos_pairs(c_cond)
            positive_ypos_samples ,negative_ypos_samples = self.get_sample_oracle_ypos_pairs(c_cond)
            postive_pairs = torch.cat((positive_shape_samples,positive_size_samples,positive_orient_samples,positive_xpos_samples,positive_ypos_samples),dim=0).to(self.device)
            negative_pairs = torch.cat((negative_shape_samples, negative_size_samples,negative_orient_samples,negative_xpos_samples,negative_ypos_samples),dim=0).to(self.device)

            fake_x = model.decoder(z)
            total_images = torch.cat((fake_x,postive_pairs,negative_pairs),dim=0)
            latent_code, prob_fake  = model.encoder(total_images)

            g_loss = adversarial_loss(prob_fake[:64], label_real)
            cont_loss = criterionQ_con(c_cond, latent_code[:64])
            similarity_loss_shape = similarity_loss(latent_code[:64,0].view(-1,1),latent_code[64:128,0].view(-1,1) ,latent_code[384:448,0].view(-1,1) )
            similarity_loss_size = similarity_loss(latent_code[:64,1].view(-1,1),latent_code[128:192,1].view(-1,1) ,latent_code[448:512,1].view(-1,1) )
            similarity_loss_orient = similarity_loss(latent_code[:64,2].view(-1,1), latent_code[192:256,2].view(-1,1),latent_code[512:576,2].view(-1,1) )
            similarity_loss_xpos = similarity_loss(latent_code[:64,3].view(-1,1), latent_code[256:320,3].view(-1,1),latent_code[576:640,3].view(-1,1) )
            similarity_loss_ypos = similarity_loss(latent_code[:64,4].view(-1,1), latent_code[320:384,4].view(-1,1),latent_code[640:704,4].view(-1,1) )
            G_loss = g_loss + cont_loss * 0.1 + 0.1*(similarity_loss_shape + similarity_loss_size + similarity_loss_orient + similarity_loss_xpos + similarity_loss_ypos)
            G_loss.backward()

            g_optimizer.step()

            d_optimizer.zero_grad()
            latent_code, prob_real= model.encoder(images)
            loss_real = adversarial_loss(prob_real, label_real)
            loss_real.backward()

            fake_x = model.decoder(z)
            latent_code_gen, prob_fake  = model.encoder(fake_x.detach())

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


    def get_sample_oracle_shape_pairs(self , c_cond):
        # shape_factor = c_cond[:, 0]
        # c_cond_new = torch.rand(self.config['batch_size'], self.config['latent_dim'] -1, dtype=torch.float32,
        #                     device=self.device) * 2 - 1
        # c_cond_similar = torch.cat((shape_factor.view(-1,1), c_cond_new),dim=-1)
        latent = np.digitize(c_cond[:,0].cpu(),self.shape_bins)
        latent =  [x - 1 for x in latent.tolist()]
        replace_list = []
        for current_value in latent:
            replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) + 1, 3)))
            replace_list.append(replace_value)
        # differ_shape = torch.FloatTensor([random.uniform(bins[x],bins[x+1]) for x in  replace_list]).view(-1,1).to(self.device)
        # c_cond_differ = torch.cat((differ_shape, c_cond[:,1:]),dim=-1)
        negative_samples = torch.stack([self.shape_template[int(i)] for i in replace_list]).view(-1, 1, 64, 64)
        positive_samples = torch.stack([self.shape_template[int(i)] for i in latent]).view(-1, 1, 64, 64)
        return positive_samples ,negative_samples

    def get_sample_oracle_size_pairs(self , c_cond):
        # size_factor = c_cond[:, 1]
        # c_cond_new = torch.rand(self.config['batch_size'], self.config['latent_dim'] -1, dtype=torch.float32,
        #                     device=self.device) * 2 - 1
        # c_cond_similar = torch.cat((c_cond_new[:,:1], size_factor.view(-1,1), c_cond_new[:,1:]),dim=-1)
        latent = np.digitize(c_cond[:,1].cpu(),self.size_bins)
        latent =  [x - 1 for x in latent.tolist()]

        replace_list = []
        for current_value in latent:
            replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) + 1, 6)))
            replace_list.append(replace_value)
        # differ_size = torch.FloatTensor([random.uniform(bins[x], bins[x + 1]) for x in replace_list]).view(-1, 1).to(self.device)
        # c_cond_differ = torch.cat((c_cond[:,:1] ,differ_size, c_cond[:, 2:]), dim=-1)
        negative_samples = torch.stack([self.size_template[int(i)] for i in replace_list]).view(-1, 1, 64, 64)
        positive_samples = torch.stack([self.size_template[int(i)] for i in latent]).view(-1, 1, 64, 64)
        return positive_samples ,negative_samples

    def get_sample_oracle_orient_pairs(self , c_cond):
        # orient_factor = c_cond[:, 2]
        # c_cond_new = torch.rand(self.config['batch_size'], self.config['latent_dim'] -1, dtype=torch.float32,
        #                     device=self.device) * 2 - 1
        # c_cond_similar = torch.cat((c_cond_new[:,:2], orient_factor.view(-1,1), c_cond_new[:,2:]),dim=-1)
        latent = np.digitize(c_cond[:,2].cpu(), self.orientation_bins)
        latent =  [x - 1 for x in latent.tolist()]
        replace_list = []
        for current_value in latent:
            replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) + 1, 40)))
            replace_list.append(replace_value)
        # differ_orient = torch.FloatTensor([random.uniform(bins[x], bins[x + 1]) for x in replace_list]).view(-1, 1).to(self.device)
        # c_cond_differ = torch.cat((c_cond[:,:2] ,differ_orient, c_cond[:, 3:]), dim=-1)
        negative_samples = torch.stack([self.orientation_template[int(i)] for i in replace_list]).view(-1, 1, 64, 64)
        positive_samples = torch.stack([self.orientation_template[int(i)] for i in latent]).view(-1, 1, 64, 64)
        return positive_samples ,negative_samples

    def get_sample_oracle_xpos_pairs(self , c_cond):
        # xpos_factor = c_cond[:, 3]
        # c_cond_new = torch.rand(self.config['batch_size'], self.config['latent_dim'] -1, dtype=torch.float32,
        #                     device=self.device) * 2 - 1
        # c_cond_similar = torch.cat((c_cond_new[:,:3], xpos_factor.view(-1,1), c_cond_new[:,3:]),dim=-1)
        latent = np.digitize(c_cond[:,3].cpu(),self.xpos_bins)
        latent =  [x - 1 for x in latent.tolist()]
        replace_list = []
        for current_value in latent:
            replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) + 1, 32)))
            replace_list.append(replace_value)
        # differ_xpos = torch.FloatTensor([random.uniform(bins[x], bins[x + 1]) for x in replace_list]).view(-1, 1).to(self.device)
        # c_cond_differ = torch.cat((c_cond[:,:3] ,differ_xpos, c_cond[:, 4:]), dim=-1)
        negative_samples = torch.stack([self.xpos_template[int(i)] for i in replace_list]).view(-1, 1, 64, 64)
        positive_samples = torch.stack([self.xpos_template[int(i)] for i in latent]).view(-1, 1, 64, 64)
        return positive_samples ,negative_samples

    def get_sample_oracle_ypos_pairs(self , c_cond):
        # ypos_factor = c_cond[:, 4]
        # c_cond_new = torch.rand(self.config['batch_size'], self.config['latent_dim'] -1, dtype=torch.float32,
        #                     device=self.device) * 2 - 1
        # c_cond_similar = torch.cat((c_cond_new[:,:4], ypos_factor.view(-1,1)),dim=-1)
        latent = np.digitize(c_cond[:,4].cpu(),self.ypos_bins)
        latent =  [x - 1 for x in latent.tolist()]
        replace_list = []
        for current_value in latent:
            replace_value = random.choice(list(range(int(current_value))) + list(range(int(current_value) + 1, 32)))
            replace_list.append(replace_value)
        # differ_ypos = torch.FloatTensor([random.uniform(bins[x], bins[x + 1]) for x in replace_list]).view(-1, 1).to(self.device)
        # c_cond_differ = torch.cat((c_cond[:,:4] ,differ_ypos), dim=-1)
        negative_samples = torch.stack([self.ypos_template[int(i)] for i in replace_list]).view(-1, 1, 64, 64)
        positive_samples = torch.stack([self.ypos_template[int(i)] for i in latent]).view(-1, 1, 64, 64)
        return positive_samples ,negative_samples




