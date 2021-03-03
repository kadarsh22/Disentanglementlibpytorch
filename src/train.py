import time
import os
import random
from utils import *
# from sklearn.preprocessing import train_test_split
from models.infogan import CRDiscriminator

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


            G_loss = g_loss + (cont_loss * 0.1) + (loss_cr)
            G_loss.backward()

            g_optimizer.step()

            ## encoder optimization
            d_optimizer.zero_grad()
            latent_code, prob_real = model.encoder(images)
            loss_real = adversarial_loss(prob_real, label_real)
            loss_real.backward()

            fake_x = model.decoder(z)
            latent_code_gen, prob_fake = model.encoder(fake_x.detach())

            loss_fake = adversarial_loss(prob_fake, label_fake)
            loss_fake.backward()

            d_optimizer.step()


            d_loss_summary = d_loss_summary + loss_real.item()
            g_loss_summary = g_loss_summary + G_loss.item()
            info_loss_summary = info_loss_summary  + cont_loss.item()
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

    def get_disentangled_vectors(self):
        z_noise = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                             device=self.device) * 2 - 1

        c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                            device=self.device) * 2 - 1

        shifts = torch.rand(self.config['batch_size'], 1, dtype=torch.float32,
                            device=self.device) * 2 - 1

        target_indices = torch.randint( self.config['latent_dim'],(self.config['batch_size'],1)).cuda()

        z_shift_disentangled = torch.zeros(self.config['batch_size'],self.config['latent_dim']).cuda()
        for i, (index, val) in enumerate(zip(target_indices, shifts)):
            z_shift_disentangled[i][index] += val


        c_disentangled = c_cond + z_shift_disentangled


        return c_disentangled , z_noise ,c_cond

    def get_entangled_vector(self):
        entanglement_proportion_list = [0.5, 0.75, 1]
        entanglement_proportion = random.choice(entanglement_proportion_list)
        rand_mat = torch.rand(self.config['batch_size'], self.config['latent_dim'])
        k = round(entanglement_proportion * self.config['latent_dim'])  # For the general case change 0.25 to the percentage you need
        k_th_quant = torch.topk(rand_mat, k, largest=False)[0][:, -1:]
        bool_tensor = rand_mat <= k_th_quant
        desired_tensor = torch.where(bool_tensor, torch.tensor(1), torch.tensor(0)).cuda()
        c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                            device=self.device) * 2 - 1

        c_entangled = desired_tensor * c_cond
        return c_entangled


    def train_classifier(self,model):
        adverserial_loss = torch.nn.CrossEntropyLoss()
        training_loss = []
        classifier = CRDiscriminator(dim_c_cont=2).cuda()
        cr_optimizer = torch.optim.Adam(classifier.parameters(), lr=0.002, betas=(0.9, 0.999))
        classifier.train()
        for iteration in range(5000):
            cr_optimizer.zero_grad()
            label_real = torch.full((64,), 1, dtype=torch.long, device='cuda')
            label_fake = torch.full((64,), 0, dtype=torch.long, device='cuda')
            labels = torch.cat((label_real, label_fake))
            c_disentangled , z_noise ,c_cond  = self.get_disentangled_vectors()
            c_shifted = self.get_entangled_vector()
            c_entangled = c_cond + c_shifted
            z_reference = torch.cat((z_noise,c_cond),dim=1)
            z_disentangled = torch.cat((z_noise,c_disentangled),dim=1)
            z_entangled = torch.cat((z_noise,c_entangled),dim=1)
            ref_images = model.decoder(z_reference).detach().view(-1,1,64,64)
            disentangled_images = model.decoder(z_disentangled).detach().view(-1,1,64,64)
            entangled_images = model.decoder(z_entangled).detach().view(-1,1,64,64)
            images = torch.cat((disentangled_images, entangled_images))
            ref_images = torch.cat((ref_images,ref_images))
            shuffled_indices = torch.randint(0,images.size(0), (images.size(0),))
            ref_images = ref_images[shuffled_indices]
            images = images[shuffled_indices]
            labels = labels[shuffled_indices]
            # prob_disentangle = classifier(ref_images.cuda(), disentangled_images.cuda())
            prob = classifier(ref_images.cuda(), images.cuda())
            loss_dis = adverserial_loss(prob, labels)
            loss_dis.backward()
            cr_optimizer.step()
            training_loss.append(loss_dis.item())
            if iteration % 100 == 0 and iteration != 0:
                correct = 0
                total = 0
                classifier.eval()
                with torch.no_grad():
                    for k in range(1000):
                        label_real = torch.full((64,), 1, dtype=torch.float32, device='cuda')
                        label_fake = torch.full((64,), 0, dtype=torch.float32, device='cuda')
                        c_disentangled , z_noise ,c_cond  = self.get_disentangled_vectors()
                        c_shifted = self.get_entangled_vector()
                        c_entangled = c_cond + c_shifted
                        z_reference = torch.cat((z_noise,c_cond),dim=1)
                        z_disentangled = torch.cat((z_noise,c_disentangled),dim=1)
                        z_entangled = torch.cat((z_noise,c_entangled),dim=1)
                        ref_images = model.decoder(z_reference).detach().view(-1,1,64,64)
                        disentangled_images = model.decoder(z_disentangled).detach().view(-1,1,64,64)
                        entangled_images = model.decoder(z_entangled).detach().view(-1,1,64,64)
                        prob_disentangle = classifier(ref_images.cuda(), disentangled_images.cuda())
                        prob_entangle = classifier(ref_images.cuda(), entangled_images.cuda())
                        _, predicted_dis = torch.max(prob_disentangle,1)
                        _, predicted_ent = torch.max(prob_entangle,1)
                        predicted = torch.cat((predicted_dis, predicted_ent))
                        labels = torch.cat((label_real, label_fake))

                        # Total number of labels
                        total += labels.size(0)

                        # Total correct predictions
                        correct += (predicted.view(-1) == labels.view(-1)).sum()
                    classifier.train()
                    accuracy = 100 * correct.item() / total

                    print('training loss : ', sum(training_loss) / len(training_loss), "accuracy :", accuracy)
                    training_loss = []

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
