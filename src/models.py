import copy
import functools

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

import src.net_utils as net_utils

""" Blocks """
class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, activation='relu', norm='instance', pad_type='reflection', use_bias=True, use_dropout=False):
        super(ConvBlock, self).__init__()
        self.padding = None
        if pad_type == 'reflection':
            self.padding = nn.ReflectionPad2d(padding)
        elif pad_type == 'replication':
            self.padding = nn.ReplicationPad2d(padding)
        
        self.norm = None
        if norm == 'instance':
            self.norm = nn.InstanceNorm2d(output_dim)
        elif norm == 'batch':
            self.norm = nn.BatchNorm2d(output_dim)
        elif norm == 'layer':
            self.norm = LayerNorm(output_dim)

        self.activation = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()

        self.dropout = None
        if use_dropout:
            self.dropout = nn.Dropout(0.5)
        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride, bias=use_bias)
        
    def forward(self, x):
        if self.dropout:
            x = self.dropout(x)
        x = self.conv(self.padding(x))
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x

class ResLayer(nn.Module):
    def __init__(self, channels, activation, norm, pad_type):
        super(ResLayer, self).__init__()
        res_layer = [ConvBlock(channels, channels, 3, 1, 1, activation, norm, pad_type)]
        res_layer += [ConvBlock(channels, channels, 3, 1, 1, None, norm, pad_type)]
        
        self.res_layer = nn.Sequential(*res_layer)

    def forward(self, x):
        residual = x
        x = self.res_layer(x)
        x += residual
        return x

class ResBlock(nn.Module):
    def __init__(self, nr_blocks, out_channels, activation, norm, pad_type):
        super(ResBlock, self).__init__()
        res = []
        for nr in range(0, nr_blocks):
            res += [ResLayer(out_channels, activation, norm, pad_type)]
        self.res = nn.Sequential(*res)
    
    def forward(self, x):
        x = self.res(x)
        return x

class UpSampleBlock(nn.Module):
    def __init__(self, input_channels, out_channels, kernel_size=4, stride=2, padding=1, use_dropout=True):
        super(UpSampleBlock, self).__init__()
        self.dropout = None
        self.block = nn.Sequential(
            nn.ConvTranspose2d(input_channels, out_channels, kernel_size, stride, padding, bias=nn.InstanceNorm2d),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        if use_dropout:
            self.dropout = nn.Dropout(0.5)

    def forward(self, x, shortcut=None):
        x = self.block(x)
        if self.dropout:
            x = self.dropout(x)

        if shortcut is not None:
            x = torch.cat([x, shortcut], dim=1)

        return x

""" Encoder-Decoder (s)"""
class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        # First Block
        encoder_model = [ConvBlock(config.input_channels, config.gen_filters, 7, 1, 3, activation=config.gen_activation, norm='none', pad_type=config.gen_pad_type, use_bias=config.gen_bias)]
        self.out_channels = config.gen_filters

        # Downsample Blocks
        for nr in range(config.gen_n_downsamples):
            encoder_model += [ConvBlock(self.out_channels, self.out_channels * 2, 4, 2, 1, activation=config.gen_activation, norm=config.gen_norm, pad_type=config.gen_pad_type, use_bias=config.gen_bias)]
            self.out_channels *= 2
        
        # Residual Blocks
        encoder_model += [ResBlock(config.gen_n_res, self.out_channels, activation=config.gen_activation, norm=config.gen_norm, pad_type=config.gen_pad_type)]
        
        # Sequential of all steps
        self.encoder_model = nn.Sequential(*encoder_model)
        
    def forward(self, x):
        return self.encoder_model(x)

class Decoder(nn.Module):
    def __init__(self, config, enc_out_channels):
        super(Decoder, self).__init__()

        # Residual Block
        decoder_model = [ResBlock(config.gen_n_res, enc_out_channels, config.gen_activation, config.gen_norm, config.gen_pad_type)]
        self.out_channels = enc_out_channels
        # Upsample Back
        for nr in range(config.gen_n_downsamples):
            decoder_model += [nn.Upsample(scale_factor=2),
                            ConvBlock(self.out_channels, self.out_channels // 2, 5, 1, 2, norm='instance', activation=config.gen_activation, pad_type=config.gen_pad_type, use_dropout=config.gen_dropout)]
            self.out_channels //= 2
        decoder_model += [ConvBlock(self.out_channels, config.input_channels, 7, 1, 3, norm=None, activation='tanh', pad_type=config.gen_pad_type)]
        self.decoder_model = nn.Sequential(*decoder_model)

    def forward(self, x):
        return self.decoder_model(x)

""" Generators """
class UNetGenerator(nn.Module):
    def __init__(self, config, encoder=None, decoder=None):
        super(UNetGenerator, self).__init__()
        if encoder:
            self.down = encoder
        else:
            self.down = nn.ModuleList([
                ConvBlock(config.input_channels, config.gen_filters, 4, 2, 1, 'lrelu', norm='none', use_bias=nn.InstanceNorm2d),
                ConvBlock(config.gen_filters, config.gen_filters*2, 4, 2, 1, 'lrelu', norm='instance', use_bias=nn.InstanceNorm2d),
                ConvBlock(config.gen_filters*2, config.gen_filters*4, 4, 2, 1, 'lrelu', norm='instance', use_bias=nn.InstanceNorm2d),
                ConvBlock(config.gen_filters*4, config.gen_filters*8, 4, 2, 1, 'lrelu', norm='instance', use_bias=nn.InstanceNorm2d),
                ConvBlock(config.gen_filters*8, config.gen_filters*8, 4, 2, 1, 'lrelu', norm='instance', use_bias=nn.InstanceNorm2d),
            ])
        if decoder:
            self.up = decoder
        else:
            self.up = nn.ModuleList([
                UpSampleBlock(config.gen_filters*8, config.gen_filters*8),
                UpSampleBlock(config.gen_filters*16, config.gen_filters*4, use_dropout=False),
                UpSampleBlock(config.gen_filters*8, config.gen_filters*2, use_dropout=False),
                UpSampleBlock(config.gen_filters*4, config.gen_filters, use_dropout=False),
            ])

        self.model = nn.Sequential(
            nn.ConvTranspose2d(config.gen_filters * 2, 3, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        skips = []
        for l in self.down:
            x = l(x)
            skips.append(x)

        skips = reversed(skips[:-1])
        for l, s in zip(self.up, skips):
            x = l(x, s)

        return self.model(x)

class VAEGen(nn.Module):
    def __init__(self, config, encoder):
        super(VAEGen, self).__init__()
        
        # For shared weights
        if encoder:
            self.encoder = encoder
        else:
            self.encoder = Encoder(config)
        self.decoder = Decoder(config, self.encoder.out_channels)
            

    def forward(self, x):
        # This code assumes std = matrix(1), mean = embedding
        # Encode input
        embedding = self.encoder(x)
        if self.training == True:
            noise = net_utils.cuda(Variable(torch.randn(embedding.size(), requires_grad=True)))
            result = self.decoder(embedding+noise)
        # Don't add noise when testing
        else:
            result = self.decoder(embedding)
        return result

""" Discriminators """
class SimpleDiscriminator(nn.Module):
    def __init__(self, config):
        super(SimpleDiscriminator, self).__init__()
        
        self.block = nn.Sequential(
            ConvBlock(config.input_channels, config.gen_filters, 4, 2, 1, 'lrelu', norm='none'),
            ConvBlock(config.gen_filters, config.gen_filters*2, 4, 2, 1, 'lrelu', norm='instance'),
            ConvBlock(config.gen_filters*2, config.gen_filters*4, 4, 2, 1, 'lrelu', norm='instance'),
            ConvBlock(config.gen_filters*4, config.gen_filters*8, 4, 1, 1, 'lrelu', norm='instance'),
        )

        self.conv = nn.Conv2d(config.gen_filters*8, 1, 4, 1, 1)
    
    def forward(self, x):
        x = self.block(x)
        return self.conv(x)

class PixelDiscriminator(nn.Module):
    def __init__(self, config):
        super(PixelDiscriminator, self).__init__()
        dis_model = [
            nn.Conv2d(config.input_channels, config.dis_filters, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(config.dis_filters, config.dis_filters * 2, kernel_size=1, stride=1, padding=0, bias=config.dis_bias),
            norm_layer(config.dis_filters * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(config.dis_filters * 2, 1, kernel_size=1, stride=1, padding=0, bias=config.dis_bias)]

        self.dis_model = nn.Sequential(*dis_model)

    def forward(self, input):
        return self.dis_model(input)

class Discriminator(nn.Module):
    def __init__(self, config):
        super(Discriminator, self).__init__()

        model = [ConvBlock(config.input_channels, config.dis_filters, 4, 2, 1, config.dis_activation, config.dis_norm, config.dis_pad_type)]
        self.out_channels = config.dis_filters
        for nr in range(config.dis_n_layer):
            model += [ConvBlock(self.out_channels, self.out_channels*2, 4, 2, 1, config.dis_activation, config.dis_norm, config.dis_pad_type)]
            self.out_channels *= 2
        model += [nn.Conv2d(self.out_channels, 1, 1, 1, 0)]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)

""" Model Utilities """
class LambdaLR():
    def __init__(self, epochs, offset, decay_epoch):
        self.epochs = epochs
        self.offset = offset
        self.decay_epoch = decay_epoch

    def step(self, epoch):
        if (self.epochs > self.decay_epoch):
            return 1.0 - max(0, epoch + self.offset - self.decay_epoch)/(self.epochs - self.decay_epoch)
        return 0

class SampleFromGenerated():
    """ 
        Strategy introduced by Shrivastava et al.
        Sample from generated images during training.
        Args:
        max_items: 50 (default)
    """
    def __init__(self, max_items=50):
        self.max_elems = max_items
        self.elems = 0
        self.items = []
    def __call__(self, items):
        sample = []
        for item in items:
            if self.elems < self.max_elems:
                self.items.append(item)
                self.elems += 1
                sample.append(item)
            else:
                if np.random.ranf() > 0.5:
                    idx = np.random.randint(0, self.max_elems)
                    tmp = copy.copy(self.items[idx])
                    self.items[idx] = item
                    sample.append(tmp)
                else:
                    sample.append(item)
        return sample

class LayerNorm(nn.Module):
    def __init__(self, num_features, eps=1e-5, affine=True):
        super(LayerNorm, self).__init__()
        self.num_features = num_features
        self.affine = affine
        self.eps = eps

        if self.affine:
            self.gamma = nn.Parameter(torch.Tensor(num_features).uniform_())
            self.beta = nn.Parameter(torch.zeros(num_features))

    def forward(self, x):
        shape = [-1] + [1] * (x.dim() - 1)
        if x.size(0) == 1:
            mean = x.view(-1).mean().view(*shape)
            std = x.view(-1).std().view(*shape)
        else:
            mean = x.view(x.size(0), -1).mean(1).view(*shape)
            std = x.view(x.size(0), -1).std(1).view(*shape)

        x = (x - mean) / (std + self.eps)

        if self.affine:
            shape = [1, -1] + [1] * (x.dim() - 2)
            x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x