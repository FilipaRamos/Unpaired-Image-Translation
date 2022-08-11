import os
import glob
import torch
import functools

from torch import nn
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

import numpy as np
import matplotlib.image as mpimg
from matplotlib import pyplot as plt
from torchvision.utils import make_grid

# To save the checkpoint 
def save_checkpoint(state, save_path):
    torch.save(state, save_path)

# To load the checkpoint
def load_checkpoint(ckpt_path, map_location=None):
    ckpt = torch.load(ckpt_path, map_location=map_location)
    print('[*] Loading checkpoint from %s succeed!' % ckpt_path)
    return ckpt
                
# Tensor sample shape: B x C x H x W
def show_tensor_sample(tensor_s):
    viz = make_grid(tensor_s, nrow=6, padding=2).permute(1, 2, 0).detach().numpy()
    viz = viz * 0.5 + 0.5
    viz = viz * 255.0
    viz = viz.astype(int)
    
    fig = plt.figure(figsize=(18,8), facecolor='w')
    plt.imshow(viz)
    plt.axis('off')
    plt.title('Photo')
    plt.show()

def save_epoch_sample_results(epoch, stats_dir, photo, monet):
    photo = photo[0]
    monet = monet[0]

    # Reverse the normalization
    photo = photo * 0.5 + 0.5
    photo = photo * 255
    monet = monet * 0.5 + 0.5
    monet = monet * 255

    # Create grid
    temp = make_grid([photo, monet], nrow=2, padding=1).permute(1, 2, 0).detach().cpu().numpy()
    temp = temp.astype(int)
    
    fig = plt.figure(figsize=(18, 8), facecolor='w')
    plt.imshow(temp)
    plt.axis('off')
    plt.title('Photo')
    stats_dis = os.path.join(stats_dir, 'epochs')
    path = os.path.join(stats_dir, 'sample_results_epoch_{}.png'.format(epoch))
    fig.savefig(path)   # save the figure to file
    plt.close(fig)

def save_loss_history(gen_loss_photo_history, gen_loss_monet_history, id_loss_photo_history, id_loss_monet_history, cyc_loss_history, photo_dis_loss_history, monet_dis_loss_history, checkpoint, stats_dir):
    filename = checkpoint.split('.')[0]

    fig, axes = plt.subplots(ncols=1, nrows=4, figsize=(18, 12), facecolor='w')
    nr_samples = len(gen_loss_photo_history)
    gen_loss_monet = np.array(gen_loss_monet_history)+np.array(id_loss_monet_history)+np.array(cyc_loss_history)
    gen_loss_photo = np.array(gen_loss_photo_history)+np.array(id_loss_photo_history)+np.array(cyc_loss_history)

    axes[0].plot(np.arange(nr_samples), list(gen_loss_photo), label='generator loss photo')
    axes[0].plot(np.arange(nr_samples), photo_dis_loss_history, label='discriminator loss photo')
    axes[0].legend()
    axes[0].set_xlabel('Epoch')

    axes[1].plot(np.arange(nr_samples), list(gen_loss_monet), label='generator loss monet')
    axes[1].plot(np.arange(nr_samples), monet_dis_loss_history, label='discriminator loss monet')
    axes[1].legend()
    axes[1].set_xlabel('Epoch')

    axes[2].plot(np.arange(nr_samples), gen_loss_photo_history, label='gen loss photo')
    axes[2].plot(np.arange(nr_samples), id_loss_photo_history, label='id loss photo')
    axes[2].plot(np.arange(nr_samples), cyc_loss_history, label='cyc loss')
    axes[2].legend()
    axes[2].set_xlabel('Epoch')

    axes[3].plot(np.arange(nr_samples), gen_loss_monet_history, label='gen loss monet')
    axes[3].plot(np.arange(nr_samples), id_loss_monet_history, label='id loss monet')
    axes[3].plot(np.arange(nr_samples), cyc_loss_history, label='cyc loss')
    axes[3].legend()
    axes[3].set_xlabel('Epoch')

    path = os.path.join(stats_dir, 'loss_history_{}.png'.format(filename))
    fig.savefig(path)   # save the figure to file
    plt.close(fig)    # close the figure window