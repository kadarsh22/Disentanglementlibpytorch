import time
import os
import random
from utils import *
from sklearn.model_selection import train_test_split
from contrastive_model import InfonceEncoder
import os
import itertools
torch.autograd.set_detect_anomaly(True)

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

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
        info_optimizer = optimizer[2]
        start_time = time.time()
        d_loss_summary, g_loss_summary, info_loss_summary, mig_loss_summary = 0, 0, 0, 0
        model.encoder.to(self.device)
        model.decoder.to(self.device)

        adversarial_loss = torch.nn.BCELoss()
        criterionQ_con = log_gaussian()
        label_real = torch.full((self.config['batch_size'],), 1, dtype=torch.float32, device=self.device)
        label_fake = torch.full((self.config['batch_size'],), 0, dtype=torch.float32, device=self.device)
        cr_labels = list(itertools.chain.from_iterable(itertools.repeat(x, 64) for x in range(5)))
        cr_labels_tensor = torch.LongTensor(cr_labels).to(self.device)
        cross_entropy = torch.nn.CrossEntropyLoss()

        for iter, images in enumerate(self.train_loader):

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

            G_loss = g_loss + cont_loss * 0.05
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

            if iter % 500 == 0:
                network_one, network_two, network_three, network_four, network_five = self.train_infonce(model)

            info_optimizer.zero_grad()

            z_noise = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                                 device=self.device) * 2 - 1
            c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                                device=self.device) * 2 - 1

            z = torch.cat((z_noise, c_cond), dim=1)

            fake_x = model.decoder(z)
            latent_code_gen, prob_fake = model.encoder.forward_no_spectral(fake_x)


            cr_inputs = torch.cat((latent_code_gen.view(-1, 1), c_cond.view(-1, 1)), dim=1).view(-1, 5, 2)
            net_zero_out = network_one(cr_inputs[:, 0, :])
            net_one_out = network_two(cr_inputs[:, 1, :])
            net_two_out = network_three(cr_inputs[:, 2, :])
            net_three_out = network_four(cr_inputs[:, 3, :])
            net_four_out = network_five(cr_inputs[:, 4, :])

            cr_loss = cross_entropy(net_zero_out, cr_labels_tensor[:64]) + cross_entropy(net_one_out, cr_labels_tensor[64:128]) + cross_entropy(
                net_two_out, cr_labels_tensor[128:192]) + cross_entropy(net_three_out, cr_labels_tensor[192:256] ) + cross_entropy(
                net_four_out, cr_labels_tensor[256:])

            cr_loss.backward()
            info_optimizer.step()

            d_loss_summary = d_loss_summary + D_loss.item()
            g_loss_summary = g_loss_summary + G_loss.item()
            info_loss_summary = info_loss_summary + q_loss.item() + cont_loss.item()
            mig_loss_summary = mig_loss_summary +cr_loss.item()
        #
        logging.info("Epochs  %d / %d Time taken %d sec  D_Loss: %.5f, G_Loss %.5F , MIG Loss %.5F" % (
            epoch, self.config['epochs'], time.time() - start_time,
            g_loss / len(self.train_loader), d_loss_summary / len(self.train_loader),
            mig_loss_summary / len(self.train_loader)))
        self.train_hist_gan['d_loss'].append(d_loss_summary / len(self.train_loader))
        self.train_hist_gan['g_loss'].append(g_loss_summary / len(self.train_loader))
        self.train_hist_gan['info_loss'].append(info_loss_summary / len(self.train_loader))
        self.train_hist_gan['info_loss'].append(info_loss_summary / len(self.train_loader))
        return model, self.train_hist_gan, (d_optimizer, g_optimizer,info_optimizer)

    def train_infonce(self, model_gan):
        model_list = []
        for i in range(5):

            model = InfonceEncoder()
            model.cuda()
            optimizer = torch.optim.Adam(model.parameters(),lr=1e-4)
            criterion = torch.nn.CrossEntropyLoss()
            label_list = [i for i in range(5)]
            label_list.remove(i)
            full_label_list = list(itertools.chain.from_iterable(itertools.repeat(x, 64) for x in label_list))
            postive_labels = torch.LongTensor([i]*64)
            negative_labels = torch.LongTensor(full_label_list)
            labels = torch.cat((postive_labels,negative_labels)).view(-1,1).to(self.device)
            for j in range(30):
                z_noise = torch.rand(self.config['batch_size'], self.config['noise_dim'], dtype=torch.float32,
                                         device=self.device) * 2 - 1
                c_cond = torch.rand(self.config['batch_size'], self.config['latent_dim'], dtype=torch.float32,
                                        device=self.device) * 2 - 1

                z = torch.cat((z_noise, c_cond), dim=1)

                fake_x = model_gan.decoder(z)
                latent_code, _ = model_gan.encoder.forward_no_spectral(fake_x)

                positive_samples = torch.cat((latent_code.detach().view(-1, 1), c_cond.view(-1, 1)), dim=1).view(-1, 5, 2)[:,i,:]
                negative_samples_list = list(torch.unbind(c_cond,dim=1))
                negative_samples_list.pop(i)
                negative_samples = torch.stack([torch.cat((negative_samples_list[i].view(-1,1),latent_code[:,i].view(-1,1)),dim=1) for i in range(4)]).view(-1,2)
                input_tensor = torch.cat((positive_samples, negative_samples.detach()))
                dataset = NewDataset(input_tensor,labels)
                train_loader = torch.utils.data.DataLoader(dataset, batch_size=self.config['batch_size'], shuffle=True)

                for input, label_batch in train_loader:
                    optimizer.zero_grad()
                    outputs = model(input)
                    loss = criterion(outputs, label_batch.view(-1))
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

            model_list.append(model)
        return model_list[0] ,model_list[1] ,model_list[2] , model_list[4] , model_list[4]

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
