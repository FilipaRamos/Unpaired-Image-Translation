import os
import glob
import tqdm
import torch
import functools
import itertools

import src.utils as utils
import src.models as models
import src.dataset as dataset
import src.net_utils as net_utils

import numpy as np
import torch.optim as optim

from src.dataset import Transform, MonetDataLoader, MonetDataset

import torch
from torch import nn
from tqdm import trange
from PIL import Image
from torch.autograd import Variable

"""
    CycleGAN Main Processor
    Includes the training and testing cycles
    Args: argument parser, config class
"""
class CycleGAN():
    def __init__(self, args, config):
        norm_layer = net_utils.get_norm_layer(norm_type=config.norm)
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if args.gen_model == 'encoder':
            self.G_monet = models.VAEGen(config, None)
            encoder = None
            if config.enc_sw:
                encoder = self.G_monet.encoder
            self.G_photo = models.VAEGen(config, encoder)
        elif args.gen_model == 'unet':
            self.G_monet = models.UNetGenerator(config)
            encoder = None
            decoder = None
            if config.enc_sw:
                encoder = self.G_monet.down 
            if config.dec_sw:
                decoder = self.G_monet.up
            self.G_photo = models.UNetGenerator(config, encoder=encoder, decoder=decoder)
        if args.dis_model == 'normal':
            self.D_photo = models.Discriminator(config)
            self.D_monet = models.Discriminator(config)
        elif args.dis_model == 'pixel':
            self.D_photo = models.PixelDiscriminator(config)
            self.D_monet = models.PixelDiscriminator(config)
        elif args.dis_model == 'simple':
            self.D_photo = models.SimpleDiscriminator(config)
            self.D_monet = models.SimpleDiscriminator(config)

        net_utils.print_net([self.G_monet, self.G_photo, self.D_photo, self.D_monet])
        
        self.id_loss = nn.L1Loss(reduction="mean")
        self.cyc_loss = nn.L1Loss(reduction="mean")
        self.adv_loss = net_utils.GANLoss(config.loss_mode)
        
        self.gen_optimizer = torch.optim.Adam(itertools.chain(self.G_monet.parameters(), self.G_photo.parameters()), lr=args.lr, betas=(config.beta1, config.beta2))
        self.dis_optimizer = torch.optim.Adam(itertools.chain(self.D_photo.parameters(), self.D_monet.parameters()), lr=args.lr, betas=(config.beta1, config.beta2))
        
        self.gen_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.gen_optimizer, lr_lambda=models.LambdaLR(args.epochs, 0, config.decay_epoch).step)
        self.dis_lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.dis_optimizer, lr_lambda=models.LambdaLR(args.epochs, 0, config.decay_epoch).step)
        
        if not os.path.isdir(args.checkpoint_dir):
            os.mkdir(args.checkpoint_dir)

        self.gen_loss_photo_history = []
        self.gen_loss_monet_history = []
        self.id_loss_monet_history = []
        self.id_loss_photo_history = []
        self.cyc_loss_history = []
        self.monet_dis_loss_history = []
        self.photo_dis_loss_history = []
            
        # Restart training from checkpoint
        if os.path.exists(os.path.join(args.checkpoint_dir, args.checkpoint_name)):
            ckpt = utils.load_checkpoint(os.path.join(args.checkpoint_dir, args.checkpoint_name))
            self.start_epoch = ckpt['epoch']
            self.G_monet.load_state_dict(ckpt['G_photo_monet'])
            self.G_photo.load_state_dict(ckpt['G_monet_photo'])
            self.D_photo.load_state_dict(ckpt['D_photo'])
            self.D_monet.load_state_dict(ckpt['D_monet'])
            self.gen_optimizer.load_state_dict(ckpt['gen_optimizer'])
            self.dis_optimizer.load_state_dict(ckpt['dis_optimizer'])
        else:
            print('[LOG] There are no checkpoints!')
            self.start_epoch = 0

            # Initialize weights since there are no checkpoints
            net_utils.init_weights(self.G_photo, init_type=config.init_weights)
            net_utils.init_weights(self.G_monet, init_type=config.init_weights)
            net_utils.init_weights(self.D_photo, init_type=config.init_weights)
            net_utils.init_weights(self.D_monet, init_type=config.init_weights)
        
        self.config = config
        # Move networks to GPU
        net_utils.cuda(self.G_photo)
        net_utils.cuda(self.G_monet)
        net_utils.cuda(self.D_photo)
        net_utils.cuda(self.D_monet)


    def train(self, args, config, img_size=256):
        transform = Transform(img_size=img_size)
        # Create Dataloader
        dm = MonetDataLoader(args.photo_path, args.monet_path, args.batch_size, transform, mode='train')
        # Load Paths
        dm.prepare_data()
        # Get Samples
        dataloader = dm.train_dataloader()
        print("Nr of training samples: {}".format(len(dataloader.dataset)))
        
        gen_photo_sample = models.SampleFromGenerated()
        gen_monet_sample = models.SampleFromGenerated()
        
        for epoch in tqdm.tqdm(range(self.start_epoch, args.epochs)):            
            for i, (photo_sample, monet_sample) in enumerate(dataloader):
                step = epoch * len(dataloader) + i + 1
                #print("[LOG] Epoch = ", epoch, " Step = ", step)
                
                net_utils.set_grad([self.D_photo, self.D_monet], False)
                self.gen_optimizer.zero_grad()
                
                # Forward Pass                
                photo_sample = Variable(photo_sample)
                monet_sample = Variable(monet_sample)
                photo_sample, monet_sample = net_utils.cuda([photo_sample, monet_sample])

                # Photo -> Monet / Monet -> Photo
                gen_monet = self.G_monet(photo_sample)
                reconst_photo = self.G_photo(gen_monet)
                
                # Monet -> Photo / Photo -> Monet
                gen_photo = self.G_photo(monet_sample)
                reconst_monet = self.G_monet(gen_photo)
                
                # Identity Generation
                id_monet = self.G_monet(monet_sample)
                id_photo = self.G_photo(photo_sample)
                
                # Check validity
                d_fake_photo = self.D_photo(gen_photo)
                d_fake_monet = self.D_monet(gen_monet)
                
                # Loss Calculation
                real_label = net_utils.cuda(Variable(torch.ones_like(d_fake_photo)))

                m_gen_loss = self.adv_loss(d_fake_monet, True)
                p_gen_loss = self.adv_loss(d_fake_photo, True)

                cyc_loss = self.cyc_loss(reconst_monet, monet_sample) * self.config.lmb + self.cyc_loss(reconst_photo, photo_sample) * self.config.lmb

                id_loss_monet = self.id_loss(id_monet, monet_sample) * self.config.lmb * self.config.id_coef
                id_loss_photo = self.id_loss(id_photo, photo_sample) * self.config.lmb * self.config.id_coef

                total_gen_loss = m_gen_loss + p_gen_loss + cyc_loss + id_loss_monet + id_loss_photo

                total_gen_loss.backward()
                self.gen_optimizer.step()
                
                # Discriminators
                net_utils.set_grad([self.D_photo, self.D_monet], True)
                self.dis_optimizer.zero_grad()

                # Sample from previously generated samples
                gen_photo = Variable(torch.Tensor(gen_photo_sample([gen_photo.cpu().data.numpy()])[0]))
                gen_monet = Variable(torch.Tensor(gen_monet_sample([gen_monet.cpu().data.numpy()])[0]))
                gen_photo, gen_monet = net_utils.cuda([gen_photo, gen_monet])

                # Forward Pass again since we updated the discriminator
                d_real_photo = self.D_photo(photo_sample)
                d_real_monet = self.D_monet(monet_sample)

                d_fake_photo = self.D_photo(gen_photo)
                d_fake_monet = self.D_monet(gen_monet)

                # Loss Calculation
                real_label = net_utils.cuda(Variable(torch.ones_like(d_fake_monet)))
                fake_label = net_utils.cuda(Variable(torch.zeros_like(d_fake_monet)))

                real_monet_dis_loss = self.adv_loss(d_real_monet, True)
                fake_monet_dis_loss = self.adv_loss(d_fake_monet, False)

                real_photo_dis_loss = self.adv_loss(d_real_photo, True)
                fake_photo_dis_loss = self.adv_loss(d_fake_photo, False)

                if config.loss_mode == 'imp-wgan':
                    gradient_penalty_m, gradients_m = net_utils.cal_gradient_penalty(
                            self.D_monet, monet_sample, gen_monet, lambda_gp=10.0
                        )
                    gradient_penalty_m.backward(retain_graph=True)
                    gradient_penalty_p, gradients_m = net_utils.cal_gradient_penalty(
                            self.D_photo, photo_sample, gen_photo, lambda_gp=10.0
                        )
                    gradient_penalty_p.backward(retain_graph=True)

                monet_dis_loss = (real_monet_dis_loss + fake_monet_dis_loss) * 0.5 * config.penalisation
                photo_dis_loss = (real_photo_dis_loss + fake_photo_dis_loss) * 0.5
                
                monet_dis_loss.backward()
                photo_dis_loss.backward()
                self.dis_optimizer.step()
                
            # To plot loss history later
            self.gen_loss_photo_history.append(m_gen_loss)
            self.gen_loss_monet_history.append(p_gen_loss)
            self.id_loss_monet_history.append(id_loss_monet)
            self.id_loss_photo_history.append(id_loss_photo)
            self.cyc_loss_history.append(cyc_loss)
            self.monet_dis_loss_history.append(monet_dis_loss)
            self.photo_dis_loss_history.append(photo_dis_loss)

            utils.save_loss_history(self.gen_loss_photo_history, self.gen_loss_monet_history, \
            self.id_loss_photo_history, self.id_loss_monet_history, self.cyc_loss_history, \
            self.photo_dis_loss_history, self.monet_dis_loss_history, args.checkpoint_name, args.stats_dir)

            utils.save_checkpoint({'epoch': epoch + 1,
                                   'D_photo': self.D_photo.state_dict(),
                                   'D_monet': self.D_monet.state_dict(),
                                   'G_photo_monet': self.G_monet.state_dict(),
                                   'G_monet_photo': self.G_photo.state_dict(),
                                   'gen_optimizer': self.gen_optimizer.state_dict(),
                                   'dis_optimizer': self.dis_optimizer.state_dict()},
                                  os.path.join(args.checkpoint_dir, args.checkpoint_name))
            utils.save_epoch_sample_results(epoch + 1, args.stats_dir, photo_sample.clone(), gen_monet.clone())
            self.gen_lr_scheduler.step()
            self.dis_lr_scheduler.step()

    def test(self, args):
        if not os.path.isdir(args.results_dir):
            os.mkdir(args.results_dir)
        
        from PIL import Image
        photos_path = glob.glob(os.path.join(args.photo_path, '*.jpg'))
        self.G_monet.eval()

        for path in photos_path:
            photo_id = path.split('/')[-1]
            data = dataset.Transform()
            photo = data(Image.open(path), mode='test')
            photo = photo.unsqueeze(0)
            photo = net_utils.cuda(photo)
            with torch.no_grad():
                monet = self.G_monet(photo)[0]
                monet = monet * 0.5 + 0.5
                monet = monet * 255
                monet = monet.detach().cpu().numpy().astype(np.uint8)
            
            monet = np.transpose(monet, [1,2,0])
            monet = Image.fromarray(monet)
            monet.save(os.path.join(args.results_dir, photo_id), 'jpeg')
            print ("Saving image : ", photo_id)
        
        print("Done saving generated paintings.")