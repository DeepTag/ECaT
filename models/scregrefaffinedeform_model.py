import torch
import itertools
from util.image_pool import ImagePool
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from . import cyclegan_networks
from . import losses
from . import registration_net_cas_with_generator_detach


class SCREGREFAFFINEDEFORMModel(BaseModel):
    """
    This class implements the unpaired image translation model with spatially correlative loss
    """

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """
        :param parser: original options parser
        :return: the modified parser
        """
        parser.set_defaults(no_dropout=True)

        parser.add_argument('--attn_layers', type=str, default='4, 7, 9', help='compute spatial loss on which layers')
        parser.add_argument('--patch_nums', type=float, default=256, help='select how many patches for shape consistency, -1 use all')
        parser.add_argument('--stage_1_epochs', type=float, default=300, help='epochs to train the generator in stage 1')
        parser.add_argument('--stage_2_epochs', type=float, default=300, help='epochs to train the generator in stage 1')
        parser.add_argument('--patch_size', type=int, default=64, help='patch size to calculate the attention')
        parser.add_argument('--loss_mode', type=str, default='cos', help='which loss type is used, cos | l1 | info')
        parser.add_argument('--use_norm', action='store_true', help='normalize the feature map for FLSeSim')
        parser.add_argument('--learned_attn', action='store_false', help='use the learnable attention map')
        parser.add_argument('--augment', action='store_false', help='use data augmentation for contrastive learning')
        parser.add_argument('--T', type=float, default=0.07, help='temperature for similarity')
        parser.add_argument('--lambda_spatial', type=float, default=10.0, help='weight for spatially-correlative loss')
        parser.add_argument('--lambda_spatial_global', type=float, default=10.0, help='weight for spatially-correlative loss')
        parser.add_argument('--lambda_spatial_idt', type=float, default=0.0, help='weight for idt spatial loss')
        parser.add_argument('--lambda_perceptual', type=float, default=0.0, help='weight for feature consistency loss')
        parser.add_argument('--lambda_style', type=float, default=0.0, help='weight for style loss')
        parser.add_argument('--lambda_identity', type=float, default=0.0, help='use identity mapping')
        parser.add_argument('--lambda_gradient', type=float, default=0.0, help='weight for the gradient penalty')

        return parser

    def __init__(self, opt):
        """
        Initialize the translation losses
        :param opt: stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out
        self.loss_names = ['G_s', 'Reg', 'D_real', 'D_fake', 'G_GAN']
        # specify the images you want to save/display
        self.visual_names = ['real_A', 'fake_B' , 'real_B']
        # specify the models you want to save to the disk
        self.model_names = ['G', 'D', 'Reg'] if self.isTrain else ['G', 'Reg']
        # define the networks

        self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, 'ncn_splited_resnet_9blocks', opt.norm,
                                      not opt.no_dropout, opt.init_type, opt.init_gain, opt.no_antialias,
                                      opt.no_antialias_up, self.gpu_ids,
                                      opt)
        vol_size = (192, 192)
        nf_enc = [16, 32, 32, 32]
        nf_dec = [32, 32, 32, 32, 16]

        regnet = registration_net_cas_with_generator_detach.Affine_diffeo_reg_net(vol_size, nf_enc, nf_dec)
        self.netReg = cyclegan_networks.init_net(regnet, opt.init_type, opt.init_gain, self.gpu_ids)

        # define the training process
        if self.isTrain:
            self.netD = networks.define_D(opt.output_nc, opt.ndf, opt.netD, opt.n_layers_D, opt.norm,
                                          opt.init_type, opt.init_gain, opt.no_antialias, self.gpu_ids, opt)
            self.attn_layers = [int(i) for i in self.opt.attn_layers.split(',')]
            if opt.lambda_identity > 0.0 or opt.lambda_spatial_idt > 0.0:
                # only works when input and output images have the same number of channels
                self.visual_names.append('idt_B')
                if opt.lambda_identity > 0.0:
                    self.loss_names.append('idt_B')
                if opt.lambda_spatial_idt > 0.0:
                    self.loss_names.append('G_s_idt_B')
                assert (opt.input_nc == opt.output_nc)
            if opt.lambda_gradient > 0.0:
                self.loss_names.append('D_Gradient')
            self.fake_B_pool = ImagePool(opt.pool_size) # create image buffer to store previously generated images
            # define the loss function
            self.netPre = losses.VGG16().to(self.device)
            self.criterionGAN = losses.GANLoss(opt.gan_mode).to(self.device)
            self.criterionIdt = torch.nn.L1Loss()
            self.criterionStyle = losses.StyleLoss().to(self.device)
            self.criterionFeature = losses.PerceptualLoss().to(self.device)
            self.criterionSpatial = losses.SpatialCorrelativeLoss(opt.loss_mode, opt.patch_nums, opt.patch_size, opt.use_norm,
                                    opt.learned_attn, gpu_ids=self.gpu_ids, T=opt.T).to(self.device)
            self.normalization = losses.Normalization(self.device)

            # for reg
            reg_loss = registration_net_cas_with_generator_detach.Diffeo_reg_loss()
            self.criterionReg = registration_net_cas_with_generator_detach.NCC()
            self.criterionRegFlow = reg_loss.gradient_loss

            # define the contrastive loss
            if opt.learned_attn:
                self.netF = self.criterionSpatial
                self.model_names.append('F')
                self.loss_names.append('spatial')
            else:
                self.set_requires_grad([self.netPre], False)
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD.parameters()), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_Reg = torch.optim.Adam(itertools.chain(self.netReg.parameters()), lr=1e-4,betas=(opt.beta1, 0.999)) #lr=1e-4
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)
            self.optimizers.append(self.optimizer_Reg)

    def data_dependent_initialize(self, data):
        """
        The learnable spatially-correlative map is defined in terms of the shape of the intermediate, extracted features
        of a given network (encoder or pretrained VGG16). Because of this, the weights of spatial are initialized at the
        first feedforward pass with some input images
        :return:
        """
        self.set_input(data)
        bs_per_gpu = self.real_A.size(0) // max(len(self.opt.gpu_ids), 1)
        self.real_A = self.real_A[:bs_per_gpu]
        self.real_B = self.real_B[:bs_per_gpu]

        self.forward()
        if self.isTrain:
            self.backward_G()
            self.optimizer_G.zero_grad()
            if self.opt.learned_attn:
                self.optimizer_F = torch.optim.Adam([{'params': list(filter(lambda p:p.requires_grad, self.netPre.parameters())), 'lr': self.opt.lr*0.0},
                                        {'params': list(filter(lambda p:p.requires_grad, self.netF.parameters()))}],
                                         lr=self.opt.lr, betas=(self.opt.beta1, self.opt.beta2))
                self.optimizers.append(self.optimizer_F)
                self.optimizer_F.zero_grad()

    def set_input(self, input):
        """
        Unpack input data from the dataloader and perform necessary pre-processing steps
        :param input: include the data itself and its metadata information
        :return:
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if self.opt.isTrain and self.opt.augment:
            self.aug_A = input['A_aug' if AtoB else 'B_aug'].to(self.device)
            self.aug_B = input['B_aug' if AtoB else 'A_aug'].to(self.device)


    def normalization_for_reg(self, img):
        v_max = torch.max(img)
        v_min = torch.min(img)
        return (img - v_min) / (v_max - v_min + torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))

    def forward(self):
        """Run forward pass"""
        # image to image translation and registration
        self.real_A = self.normalization_for_reg(self.real_A) # TAGGING
        self.real_B = self.normalization_for_reg(self.real_B) # CINE
        self.warped_coarse_B, self.flow_B_A_up, self.fake_B, self.corr_warped_cine, self.corr_warped_cine_cycle, self.corr_flow, self.warped_B, self.flow_B_A, self.warped_B_lag, self.flow_B_A_lag = self.netReg(self.real_B, self.real_A,  self.netG, detach=False)


    def backward_F(self):
        """
        Calculate the contrastive loss for learned spatially-correlative loss
        """
        norm_real_A, norm_real_B, norm_fake_B = self.normalization((self.real_A + 1) * 0.5), self.normalization((self.real_B + 1) * 0.5), self.normalization((self.fake_B.detach() + 1) * 0.5)
        if self.opt.augment:
            norm_aug_A, norm_aug_B = self.normalization((self.aug_A + 1) * 0.5), self.normalization((self.aug_B + 1) * 0.5)
            norm_real_A = torch.cat([norm_real_A, norm_real_A], dim=0)
            norm_fake_B = torch.cat([norm_fake_B, norm_aug_A], dim=0)
            norm_real_B = torch.cat([norm_real_B, norm_aug_B], dim=0)
        self.loss_spatial = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, norm_real_B)

        self.loss_spatial.backward()

    def backward_D_basic(self, netD, real, fake):
        """
        Calculate GAN loss for the discriminator
        :param netD: the discriminator D
        :param real: real images
        :param fake: images generated by a generator
        :return: discriminator loss
        """
        # real
        real.requires_grad_()
        pred_real = netD(real)
        self.loss_D_real = self.criterionGAN(pred_real, True, is_dis=True)
        # fake
        pred_fake = netD(fake)
        self.loss_D_fake = self.criterionGAN(pred_fake, False, is_dis=True)
        # combined loss
        loss_D = (self.loss_D_real + self.loss_D_fake) * 0.5
        # gradient penalty
        if self.opt.lambda_gradient > 0.0:
            self.loss_D_Gradient, _ = losses.cal_gradient_penalty(netD, real, fake, real.device, lambda_gp=self.opt.lambda_gradient)#
            loss_D += self.loss_D_Gradient
        loss_D.backward()

        return loss_D

    def backward_D(self):
        """Calculate the GAN loss for discriminator"""
        fake_B = self.fake_B_pool.query(self.fake_B)
        self.loss_D_A = self.backward_D_basic(self.netD, self.real_B, fake_B.detach())
        # self.loss_D_A = self.backward_D_basic(self.netD, self.warped_coarse_B, fake_B.detach())

    def backward_G(self):
        """Calculate the loss for generator G_A"""
        # GAN loss
        self.loss_G_GAN = 0
        self.loss_G_GAN = self.criterionGAN(self.netD(self.fake_B), True)
        # different structural loss
        self.loss_G_s = 0
        norm_real_A = self.normalization((self.real_A + 1) * 0.5)
        norm_fake_B = self.normalization((self.fake_B + 1) * 0.5)
        loss_G_s = self.Spatial_Loss(self.netPre, norm_real_A, norm_fake_B, None)
        self.loss_G_s += loss_G_s * self.opt.lambda_spatial

        self.loss_G = self.loss_G_GAN  + self.loss_G_s #+ \
                      # 0.2 * self.criterionIdt(self.fake_B, self.corr_warped_cine)
                      # +0.01 * self.criterionRegFlow(self.corr_flow)
                      # +0.2 * self.criterionIdt(self.real_B, self.corr_warped_cine_cycle)

        self.loss_G.backward()


    def backward_Reg(self):
        """Calculate the loss for Diffeo_reg_net"""
        warped_coarse_B_reg, flow_B_A_up_reg, fake_B_reg, _, _, _, warped_B_reg, flow_B_A_reg, warped_B_lag_reg, flow_B_A_lag_reg = self.netReg(self.real_B, self.real_A, self.netG, detach=True)

        a = 5 
        b = 1

        loss_smooth_flow = self.criterionRegFlow(flow_B_A_reg)

        loss_img = self.criterionReg(warped_B_reg, fake_B_reg)

        self.loss_Reg = b * loss_img + a * loss_smooth_flow

        self.loss_Reg.backward()

    def optimize_parameters(self, epoch):
        """Calculate losses, gradients, and update network weights"""
        # forward
        self.forward()
        if epoch < self.opt.stage_1_epochs:
            # print('only train I2I network')
            if self.opt.learned_attn:
                self.set_requires_grad([self.netF, self.netPre], True)
                self.optimizer_F.zero_grad()
                self.backward_F()
                self.optimizer_F.step()

            # D_A
            self.set_requires_grad([self.netD], True)
            self.optimizer_D.zero_grad()
            self.backward_D()
            self.optimizer_D.step()

            # G_A
            self.set_requires_grad([self.netD], False)
            self.optimizer_G.zero_grad()
            if self.opt.learned_attn:
                self.set_requires_grad([self.netF, self.netPre], False)
            self.backward_G()
            self.optimizer_G.step()
            self.loss_Reg = 0.0
        else:
            # print('only train Registration network')
            self.loss_spatial = 0.0
            self.loss_G_s = 0.0
            self.loss_G_GAN = 0.0
            self.loss_D_real = 0.0
            self.loss_D_fake = 0.0
            # Diffeo_Reg
            self.optimizer_Reg.zero_grad()
            self.backward_Reg()
            self.optimizer_Reg.step()

    def Spatial_Loss(self, net, src, tgt, other=None):
        """given the source and target images to calculate the spatial similarity and dissimilarity loss"""
        n_layers = len(self.attn_layers)
        feats_src = net(src, self.attn_layers, encode_only=True)
        feats_tgt = net(tgt, self.attn_layers, encode_only=True)
        if other is not None:
            feats_oth = net(torch.flip(other, [2, 3]), self.attn_layers, encode_only=True)
        else:
            feats_oth = [None for _ in range(n_layers)]

        total_loss = 0.0
        for i, (feat_src, feat_tgt, feat_oth) in enumerate(zip(feats_src, feats_tgt, feats_oth)):
            loss = self.criterionSpatial.loss(feat_src, feat_tgt, feat_oth, i)
            total_loss += loss.mean()

        if not self.criterionSpatial.conv_init:
            self.criterionSpatial.update_init_()

        return total_loss / n_layers

    def get_transfering_results(self):
        return self.real_A, self.fake_B, self.corr_warped_cine, self.warped_coarse_B, self.warped_B, self.warped_B_lag, self.real_B

    def get_eval_results(self):
        return self.fake_B, self.warped_B_lag, self.flow_B_A_lag

