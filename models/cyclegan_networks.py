"""
The network architectures is based on the implementation of CycleGAN and CUT
Original PyTorch repo of CycleGAN: https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix
Original PyTorch repo of CUT: https://github.com/taesungp/contrastive-unpaired-translation
Original CycleGAN paper: https://arxiv.org/pdf/1703.10593.pdf
Original CUT paper: https://arxiv.org/pdf/2007.15651.pdf
We use the network architecture for our default modal image translation
"""
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import functools
import numpy as np
from torch.nn import init
import math
import torch.nn.utils.spectral_norm as spectral_norm
from models.normalization import SPADE, equal_lr

##################################################################################
# Discriminator
##################################################################################
class D_NLayersMulti(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3,
                 norm_layer=nn.BatchNorm2d, num_D=1):
        super(D_NLayersMulti, self).__init__()
        # st()
        self.num_D = num_D
        if num_D == 1:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.model = nn.Sequential(*layers)
        else:
            layers = self.get_layers(input_nc, ndf, n_layers, norm_layer)
            self.add_module("model_0", nn.Sequential(*layers))
            self.down = nn.AvgPool2d(3, stride=2, padding=[
                                     1, 1], count_include_pad=False)
            for i in range(1, num_D):
                ndf_i = int(round(ndf / (2**i)))
                layers = self.get_layers(input_nc, ndf_i, n_layers, norm_layer)
                self.add_module("model_%d" % i, nn.Sequential(*layers))

    def get_layers(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw,
                              stride=2, padding=padw), nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1,
                               kernel_size=kw, stride=1, padding=padw)]

        return sequence

    def forward(self, input):
        if self.num_D == 1:
            return self.model(input)
        result = []
        down = input
        for i in range(self.num_D):
            model = getattr(self, "model_%d" % i)
            result.append(model(down))
            if i != self.num_D - 1:
                down = self.down(down)
        return result


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, no_antialias=False):
        """Construct a PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            n_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        if no_antialias:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        else:
            sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=1, padding=padw), nn.LeakyReLU(0.2, True),
                        Downsample(ndf)
                        # nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
                    ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            if no_antialias:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True)]
            else:
                sequence += [
                    nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
                    norm_layer(ndf * nf_mult),
                    nn.LeakyReLU(0.2, True),
                    Downsample(ndf * nf_mult)
                    # nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)
                ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        """Standard forward."""
        return self.model(input)


class PixelDiscriminator(nn.Module):
    """Defines a 1x1 PatchGAN discriminator (pixelGAN)"""

    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d):
        """Construct a 1x1 PatchGAN discriminator

        Parameters:
            input_nc (int)  -- the number of channels in input images
            ndf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer
        """
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        """Standard forward."""
        return self.net(input)


##################################################################################
# Generator
##################################################################################
class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6,
                 padding_type='reflect',  no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)
                          # nn.AvgPool2d(kernel_size=2, stride=2)
                        ]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                                  use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                model += [
                    Upsample(ngf * mult),
                    # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, x, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            out = self.model(x)
            return out, None

class WTA_scale(torch.autograd.Function):
    """
    We can implement our own custom autograd Functions by subclassing
    torch.autograd.Function and implementing the forward and backward passes
    which operate on Tensors.
    """

    @staticmethod
    def forward(ctx, input, scale=1e-4):
        """
        In the forward pass we receive a Tensor containing the input and return a
        Tensor containing the output. You can cache arbitrary Tensors for use in the
        backward pass using the save_for_backward method.
        """
        activation_max, index_max = torch.max(input, -1, keepdim=True)
        input_scale = input * scale  # default: 1e-4
        # input_scale = input * scale  # default: 1e-4
        output_max_scale = torch.where(input == activation_max, input, input_scale)

        mask = (input == activation_max).type(torch.float)
        ctx.save_for_backward(input, mask)
        return output_max_scale

    @staticmethod
    def backward(ctx, grad_output):
        """
        In the backward pass we receive a Tensor containing the gradient of the loss
        with respect to the output, and we need to compute the gradient of the loss
        with respect to the input.
        """
        # import pdb
        # pdb.set_trace()
        input, mask = ctx.saved_tensors
        mask_ones = torch.ones_like(mask)
        mask_small_ones = torch.ones_like(mask) * 1e-4
        # mask_small_ones = torch.ones_like(mask) * 1e-4

        grad_scale = torch.where(mask == 1, mask_ones, mask_small_ones)
        grad_input = grad_output.clone() * grad_scale
        return grad_input, None

def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)

class NCNWarpNet(nn.Module):
    """ input is Al, Bl, channel = 1, range~[0,255] """

    def __init__(self):
        super(NCNWarpNet, self).__init__()
        self.upsampling = nn.Upsample(scale_factor=4)
        self.downsampling = nn.Upsample(scale_factor=4)
        self.down_avg = nn.AvgPool2d(kernel_size=2, stride=2, padding=0, count_include_pad=False)

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H, W).to(img.device)
        coords1 = coords_grid(N, H, W).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1
        
    def grids(self, x):
        # mesh grid
        B, C, H, W = x.size()
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        if x.is_cuda:
            grid = grid.cuda()
        return grid
    

    def warp(self, x, flo):
        """
        warp an image/tensor (im2) back to im1, according to the optical flow
        x: [B, C, H, W] (im2)
        flo: [B, 2, H, W] flow
        """
        B, C, H, W = x.size()
        # mesh grid
        """
        xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
        yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
        xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
        yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
        grid = torch.cat((xx, yy), 1).float()
        
        if x.is_cuda:
            grid = grid.cuda()
        """
        grid = self.grids(x)
        vgrid = Variable(grid) + flo

        # scale grid to [-1,1]
        vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
        vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0

        vgrid = vgrid.permute(0, 2, 3, 1)
        return nn.functional.grid_sample(x, vgrid)

    def forward(
        self,
        x_feats, # tagging
        y_feats, # cine
        temperature=0.0001 * 5,
        detach_flag=False,
        WTA_scale_weight=1,
        y_img=None
    ):
        batch_size = x_feats.shape[0] # 1
        channel = x_feats.shape[1] # 256
        feature_height = x_feats.shape[2] # 48
        feature_width = x_feats.shape[3] # 48

        # pairwise cosine similarity
        theta = x_feats.view(batch_size, channel, -1)  # 1*256*(feature_height*feature_width)
        theta = theta - theta.mean(dim=-1, keepdim=True)  # center the feature
        theta_norm = torch.norm(theta, 2, 1, keepdim=True) + sys.float_info.epsilon
        theta = torch.div(theta, theta_norm)

        phi = y_feats.view(batch_size, channel, -1)  # 1*256*(feature_height*feature_width)
        phi = phi - phi.mean(dim=-1, keepdim=True)  # center the feature
        phi_norm = torch.norm(phi, 2, 1, keepdim=True) + sys.float_info.epsilon
        phi = torch.div(phi, phi_norm)

        theta_permute = theta.permute(0, 2, 1)  # 1*(feature_height*feature_width)*256
        f = torch.matmul(theta_permute, phi)  # 1*(feature_height*feature_width)*(feature_height*feature_width)

        theta = theta.view(-1, channel, feature_height, feature_width)
        phi = phi.view(-1, channel, feature_height, feature_width)
        loss_feat_pair = F.l1_loss(theta, phi)


        if detach_flag:
            f = f.detach()

        f_similarity = f.unsqueeze_(dim=1)
        similarity_map = torch.max(f_similarity, -1, keepdim=True)[0]
        similarity_map = similarity_map.view(batch_size, 1, feature_height, feature_width)

        # f can be negative
        f_WTA = f if WTA_scale_weight == 1 else WTA_scale.apply(f, WTA_scale_weight) # 1*1*2304*2304
        f_WTA = f_WTA / temperature
        f_div_C = F.softmax(f_WTA.squeeze_(dim=1), dim=-1)  # 1*2304*2304;


        B_lab = y_feats.view(batch_size, channel, -1)
        B_lab = B_lab.permute(0, 2, 1)  # 1*2304*channel

        # multiply the corr map with color
        y = torch.matmul(f_div_C, B_lab)  # 1*2304*channel
        y = y.permute(0, 2, 1).contiguous()
        y = y.view(batch_size, channel, feature_height, feature_width)  # 1*256*48*48

        similarity_map = self.upsampling(similarity_map)

        coords0, coords1 = self.initialize_flow(x_feats)
        
            
        grid = self.grids(x_feats)
        xc = grid[:,0:1,::].view(batch_size, 1, -1)
        yc = grid[:,1:2,::].view(batch_size, 1, -1)
        
        coords_x = (f_div_C * xc).sum(dim=-1)
        coords_y = (f_div_C * yc).sum(dim=-1)

        coords1 = torch.stack([coords_x, coords_y], dim=1) # 1*2*48*48
        coords1 = coords1.view(batch_size, -1, feature_height, feature_width)
        
        flow = coords1 - coords0 # 1*2*48*48

        # if y_img is not None:
        #     y_img = self.down_avg(self.down_avg(y_img))
        #     y_img = self.warp(y_img, flow)
        #     y_img = self.upsampling(y_img)

        flow = self.upsampling(flow)
        if y_img is not None:
            y_img = self.down_avg(self.down_avg(y_img))
            batch_size = y_img.shape[0]
            channel = y_img.shape[1]
            feature_height = y_img.shape[2]
            feature_width = y_img.shape[3]
            B_lab = y_img.view(batch_size, channel, -1)
            B_lab = B_lab.permute(0, 2, 1)  # 1*2304*1
            y_img = torch.matmul(f_div_C, B_lab)  # 1*2304*1

            f_div_C_v = F.softmax(f_WTA.transpose(1, 2), dim=-1)  # 1*2304*2304;
            # print('f_div_C_v.shape')
            # print(f_div_C_v.shape)
            y_img_cyc = torch.matmul(f_div_C_v, y_img).permute(0, 2, 1).contiguous()  # 1*1*2304
            y_img_cyc = y_img_cyc.view(batch_size, channel, feature_height, feature_width)  # 1*1*48*48
            y_img = y_img.permute(0, 2, 1).contiguous()
            y_img = y_img.view(batch_size, channel, feature_height, feature_width)  # 1*1*48*48
            y_img = self.upsampling(y_img)
            y_img_cyc = self.upsampling(y_img_cyc)
        else:
            y_img = None
            y_img_cyc = None

        return y, flow, y_img, y_img_cyc, loss_feat_pair


class NCNSplitedResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False,
                 n_blocks=6,
                 padding_type='reflect', no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert (n_blocks >= 0)
        super(NCNSplitedResnetGenerator, self).__init__()
        self.opt = opt
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model_enc = [nn.ReflectionPad2d(3),
                     nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                     norm_layer(ngf),
                     nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if no_antialias:
                model_enc += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                              norm_layer(ngf * mult * 2),
                              nn.ReLU(True)]
            else:
                model_enc += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                              norm_layer(ngf * mult * 2),
                              nn.ReLU(True),
                              Downsample(ngf * mult * 2)
                              # nn.AvgPool2d(kernel_size=2, stride=2)
                              ]

        mult = 2 ** n_downsampling

        for i in range(n_blocks):  # add ResNet blocks
            model_enc += [
                ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
                            use_bias=use_bias)]
        """
        model_dec = [nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1),
                     nn.ReLU(),
                     nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1)]
        """
        model_dec = [nn.Conv2d(2 * ngf * mult, ngf * mult, kernel_size=1),
                     nn.ReLU(),
                     nn.Conv2d(ngf * mult, ngf * mult, kernel_size=1)]
        #"""
        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model_dec += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                                 output_padding=1, bias=use_bias),
                              norm_layer(int(ngf * mult / 2)),
                              nn.ReLU(True)]
            else:
                model_dec += [
                    Upsample(ngf * mult),
                    # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    nn.Conv2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=1, padding=1, bias=use_bias),
                    norm_layer(int(ngf * mult / 2)),
                    nn.ReLU(True)]
        model_dec += [nn.ReflectionPad2d(3)]
        model_dec += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_dec += [nn.Tanh()]

        self.model_enc = nn.Sequential(*model_enc)
        self.model_dec = nn.Sequential(*model_dec)
        self.warp = NCNWarpNet()

    def feature_normalize(self, feature_in):
        feature_in_norm = torch.norm(feature_in, 2, 1, keepdim=True) + sys.float_info.epsilon
        feature_in_norm = torch.div(feature_in, feature_in_norm)
        return feature_in_norm

    def forward(self, x, y):
        # x: tagging y: cine (SR)
        x_enc = self.model_enc(x)
        y_enc = self.model_enc(y)

        y_enc, flow, warped_y_img, y_img_cyc, loss_feat_pair = self.warp(x_enc, y_enc, y_img=y)

        out = self.model_dec(torch.cat([x_enc, y_enc], dim=1))

        return out, warped_y_img, y_img_cyc, loss_feat_pair, flow

##################################################################################
class SpadeResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, init_type, init_gain, gpu_ids, ngf=64, norm_layer=nn.BatchNorm2d, n_blocks=6,
                 padding_type='reflect',  no_antialias=False, no_antialias_up=False, opt=None):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(SpadeResnetGenerator, self).__init__()
        self.opt = opt
        self.n_blocks = n_blocks
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            if no_antialias:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True)]
            else:
                model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=1, padding=1, bias=use_bias),
                          norm_layer(ngf * mult * 2),
                          nn.ReLU(True),
                          Downsample(ngf * mult * 2)
                          # nn.AvgPool2d(kernel_size=2, stride=2)
                        ]

        mult = 2 ** n_downsampling

        spb = SPADEResnetBlock(ngf * mult, ngf * mult, opt, init_type, init_gain, gpu_ids)
        # spb = init_net(spb, init_type, init_gain, gpu_ids, True)
        model_spade = [spb]
        # injection of the reference image features begins here
        for i in range(1, n_blocks):       # add ResNet blocks
            # model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout,
            #                       use_bias=use_bias)]
            spb = SPADEResnetBlock(ngf * mult, ngf * mult, opt, init_type, init_gain, gpu_ids)
            # spb = init_net(spb, init_type, init_gain, gpu_ids, True)
            model_spade += [spb]


        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            if no_antialias_up:
                model_spade += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1, bias=use_bias),
                          norm_layer(int(ngf * mult / 2)),
                          nn.ReLU(True)]
            else:
                up = Upsample_sp(ngf * mult)
                up = init_net(up, init_type, init_gain, gpu_ids, True)
                spb = SPADEResnetBlock(ngf * mult, int(ngf * mult / 2), opt, init_type, init_gain, gpu_ids)
                model_spade += [
                    up,
                    # nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                    spb]

        model_last = [nn.ReflectionPad2d(3)]
        model_last += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model_last += [nn.Tanh()]

        self.model = nn.Sequential(*model)
        self.model_spade = model_spade
        self.model_last = nn.Sequential(*model_last)

    def forward(self, x, seg, layers=[], encode_only=False):
        if len(layers) > 0:
            feat = x
            feats = []
            for layer_id, layer in enumerate(self.model):
                feat = layer(feat)
                if layer_id in layers:
                    feats.append(feat)
                if layer_id == layers[-1] and encode_only:
                    return None, feats
            return feat, feats
        else:
            feat = self.model(x)

            for sp_layer in self.model_spade:
                # print('feat.shape before model_spade')
                # print(feat.shape)
                # print(seg.shape)
                feat_seg_tuple = feat, seg
                feat = sp_layer(feat_seg_tuple)
                # print('feat.shape after model_spade')
                # print(feat.shape)
                # print(seg.shape)
            out = self.model_last(feat)
            return out, None

"""
fundamental functions
"""


def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes,
                             padding=0,
                             stride=strides)
    patches = unfold(images)
    return patches, paddings


##################################################################################
# Basic Blocks
##################################################################################
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

###############################################################################
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
###############################################################################
class SPADEResnetBlock(nn.Module):
    def __init__(self, fin, fout, opt, init_type, init_gain, gpu_ids, use_se=False, dilation=1):
        super().__init__()
        # Attributes
        self.learned_shortcut = (fin != fout)
        # print('line 711, self.learned_shortcut')
        # print(self.learned_shortcut)
        fmiddle = min(fin, fout)
        self.opt = opt
        self.pad_type = 'nozero'
        self.use_se = use_se

        # create conv layers
        if self.pad_type != 'zero':
            self.pad = nn.ReflectionPad2d(dilation)
            self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=0, dilation=dilation)
            self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=0, dilation=dilation)
        else:
            self.conv_0 = nn.Conv2d(fin, fmiddle, kernel_size=3, padding=dilation, dilation=dilation)
            self.conv_1 = nn.Conv2d(fmiddle, fout, kernel_size=3, padding=dilation, dilation=dilation)
        if self.learned_shortcut:
            self.conv_s = nn.Conv2d(fin, fout, kernel_size=1, bias=False)

        # apply spectral norm if specified
        if 'spectral' in opt.norm_G:
            if opt.eqlr_sn:
                self.conv_0 = equal_lr(self.conv_0)
                self.conv_1 = equal_lr(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = equal_lr(self.conv_s)
            else:
                self.conv_0 = spectral_norm(self.conv_0)
                self.conv_1 = spectral_norm(self.conv_1)
                if self.learned_shortcut:
                    self.conv_s = spectral_norm(self.conv_s)

        # define normalization layers
        spade_config_str = opt.norm_G.replace('spectral', '')
        if 'spade_ic' in opt:
            ic = opt.spade_ic
        else:
            # ic = 0 + (3 if 'warp' in opt.CBN_intype else 0) + (opt.semantic_nc if 'mask' in opt.CBN_intype else 0)
            ic = 1

        self.norm_0 = SPADE(spade_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex)

        self.norm_0 = init_net(self.norm_0, init_type, init_gain, gpu_ids, True)

        self.norm_1 = SPADE(spade_config_str, fmiddle, ic, PONO=opt.PONO, use_apex=opt.apex)
        self.norm_1 = init_net(self.norm_1, init_type, init_gain, gpu_ids, True)

        # cyclegan_networks.init_net(net, init_type, init_gain, gpu_ids, initialize_weights=('stylegan2' not in netG))

        if self.learned_shortcut:
            # print('self.learned_shortcut')
            # print(self.learned_shortcut) # True
            self.norm_s = SPADE(spade_config_str, fin, ic, PONO=opt.PONO, use_apex=opt.apex)
            self.norm_s = init_net(self.norm_s, init_type, init_gain, gpu_ids, True)

        if use_se:
            # print('use_se')
            # print(use_se) # True
            self.se_layar = SELayer(fout)
            self.se_layar = init_net(self.se_layar, init_type, init_gain, gpu_ids, True)

        for m in self.children():
            # if hasattr(m, 'init_weights'):
            init_net(m, init_type, init_gain, gpu_ids)

    # note the resnet block with SPADE also takes in |seg|,
    # the semantic segmentation map as input
    def forward(self, x_seg1):
        x, seg1 = x_seg1
        # print('x.shape')
        # print(x.shape)
        # print('seg1.shape')
        # print(seg1.shape)
        x_s = self.shortcut(x, seg1)
        if self.pad_type != 'zero':
            dx = self.conv_0(self.pad(self.actvn(self.norm_0(x, seg1))))
            dx = self.conv_1(self.pad(self.actvn(self.norm_1(dx, seg1))))
            if self.use_se:
                dx = self.se_layar(dx)
        else:
            dx = self.conv_0(self.actvn(self.norm_0(x, seg1)))
            dx = self.conv_1(self.actvn(self.norm_1(dx, seg1)))
            if self.use_se:
                dx = self.se_layar(dx)

        out = x_s + dx

        return out

    def shortcut(self, x, seg1):
        if self.learned_shortcut:
            x_s = self.conv_s(self.norm_s(x, seg1))
        else:
            x_s = x
        return x_s

    def actvn(self, x):
        return F.leaky_relu(x, 2e-1)
###############################################################################
# Helper Functions
###############################################################################
def get_filter(filt_size=3):
    if(filt_size == 1):
        a = np.array([1., ])
    elif(filt_size == 2):
        a = np.array([1., 1.])
    elif(filt_size == 3):
        a = np.array([1., 2., 1.])
    elif(filt_size == 4):
        a = np.array([1., 3., 3., 1.])
    elif(filt_size == 5):
        a = np.array([1., 4., 6., 4., 1.])
    elif(filt_size == 6):
        a = np.array([1., 5., 10., 10., 5., 1.])
    elif(filt_size == 7):
        a = np.array([1., 6., 15., 20., 15., 6., 1.])

    filt = torch.Tensor(a[:, None] * a[None, :])
    filt = filt / torch.sum(filt)

    return filt


class Downsample(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=3, stride=2, pad_off=0):
        super(Downsample, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2)), int(1. * (filt_size - 1) / 2), int(np.ceil(1. * (filt_size - 1) / 2))]
        self.pad_sizes = [pad_size + pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if(self.filt_size == 1):
            if(self.pad_off == 0):
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])


class Upsample2(nn.Module):
    def __init__(self, scale_factor, mode='nearest'):
        super().__init__()
        self.factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return torch.nn.functional.interpolate(x, scale_factor=self.factor, mode=self.mode)


class Upsample(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp):
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]



class Upsample_sp(nn.Module):
    def __init__(self, channels, pad_type='repl', filt_size=4, stride=2):
        super(Upsample_sp, self).__init__()
        self.filt_size = filt_size
        self.filt_odd = np.mod(filt_size, 2) == 1
        self.pad_size = int((filt_size - 1) / 2)
        self.stride = stride
        self.off = int((self.stride - 1) / 2.)
        self.channels = channels

        filt = get_filter(filt_size=self.filt_size) * (stride**2)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = get_pad_layer(pad_type)([1, 1, 1, 1])

    def forward(self, inp_seg):
        inp, seg = inp_seg
        # print('line 898, inp.shape')
        # print(inp.shape)
        ret_val = F.conv_transpose2d(self.pad(inp), self.filt, stride=self.stride, padding=1 + self.pad_size, groups=inp.shape[1])[:, :, 1:, 1:]
        if(self.filt_odd):
            return ret_val
        else:
            return ret_val[:, :, :-1, :-1]


def get_pad_layer(pad_type):
    if(pad_type in ['refl', 'reflect']):
        PadLayer = nn.ReflectionPad2d
    elif(pad_type in ['repl', 'replicate']):
        PadLayer = nn.ReplicationPad2d
    elif(pad_type == 'zero'):
        PadLayer = nn.ZeroPad2d
    else:
        print('Pad type [%s] not recognized' % pad_type)
    return PadLayer


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance'):
    """Return a normalization layer

    Parameters:
        norm_type (str) -- the name of the normalization layer: batch | instance | none

    For BatchNorm, we use learnable affine parameters and track running statistics (mean/stddev).
    For InstanceNorm, we do not use learnable affine parameters. We do not track running statistics.
    """
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True, track_running_stats=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'none':
        def norm_layer(x): return Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def init_weights(net, init_type='normal', init_gain=0.02, debug=False):
    """Initialize network weights.

    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.

    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if debug:
                print(classname)
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[], debug=False, initialize_weights=True):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2

    Return an initialized network.
    """
    # print(net.device)
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
    if initialize_weights:
        init_weights(net, init_type, init_gain=init_gain, debug=debug)
    # print(net.device)
    return net
