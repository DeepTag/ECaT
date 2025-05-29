import torch
import torch.nn as nn
import torch.nn.functional as nnf
import torch.nn.functional as F
import numpy as np
import math


class unet_core(nn.Module):
    """
    [unet_core] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding layers
        """
        super(unet_core, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3], dec_nf[4]))  # 5

        if self.full_size:
            self.dec.append(conv_block(dim, dec_nf[4] + 2, dec_nf[5], 1))

        if self.vm2:
            self.vm2_conv = conv_block(dim, dec_nf[5], dec_nf[6])

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x, enc_bottom=False):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)
        y = self.dec[4](y)

        # Upsample to full res, concatenate and conv
        if self.full_size:
            y = self.upsample(y)
            y = torch.cat([y, x_enc[0]], dim=1)
            y = self.dec[5](y)

        # Extra conv for vm2
        if self.vm2:
            y = self.vm2_conv(y)

        if enc_bottom:
            return y, x_enc[-3]
        return y


class unet_core2(nn.Module):
    """
    [unet_core2] is a class representing the U-Net implementation that takes in
    a fixed image and a moving image and outputs a flow-field
    """

    def __init__(self, dim, enc_nf, dec_nf, full_size=True):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding layers
        """
        super(unet_core2, self).__init__()

        self.full_size = full_size
        self.vm2 = len(dec_nf) == 7

        # Encoder functions
        self.enc = nn.ModuleList()
        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(conv_block(dim, prev_nf, enc_nf[i], 2))

        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 1
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 2
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 3
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 4
        self.dec.append(conv_block(dim, dec_nf[3]+dim, dec_nf[4]))  # 5

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # Two convs at full_size/2 res
        y = self.dec[3](y)

        # Upsample to full res, concatenate and conv
        y = self.upsample(y)
        y = torch.cat([y, x_enc[0]], dim=1)
        y = self.dec[4](y)

        return y

class vit_unet_core(nn.Module):
    """
    [vit_unet_core] is a class representing the TransUNet implementation that takes in
    a fixed image and a moving image and outputs a feature embedding.
    """

    def __init__(self, dim, enc_nf, dec_nf):
        """
        Instiatiate UNet model
            :param dim: dimension of the image passed into the net
            :param enc_nf: the number of features maps in each layer of encoding stage
            :param dec_nf: the number of features maps in each layer of decoding stage
            :param full_size: boolean value representing whether full amount of decoding layers
        """
        super(vit_unet_core, self).__init__()

        # Encoder functions
        self.enc = nn.ModuleList()

        for i in range(len(enc_nf)):
            prev_nf = 2 if i == 0 else enc_nf[i - 1]
            self.enc.append(OverlapPatchEmbedBlock(dim=enc_nf[i], num_heads=2, img_size=192/(2**i),
                                            patch_size=3,
                                            stride=2,
                                            in_chans=prev_nf,
                                            embed_dim=enc_nf[i], mlp_ratio=2,
                                         qkv_bias=False, qk_scale=None, drop=0.0,
                                         attn_drop=0.0, drop_path=0, norm_layer=nn.Identity,
                                         sr_ratio=1, linear=False))
        # Decoder functions
        self.dec = nn.ModuleList()
        self.dec.append(conv_block(dim, enc_nf[-1], dec_nf[0]))  # 0
        self.dec.append(conv_block(dim, dec_nf[0] * 2, dec_nf[1]))  # 1
        self.dec.append(conv_block(dim, dec_nf[1] * 2, dec_nf[2]))  # 2
        self.dec.append(conv_block(dim, dec_nf[2] + enc_nf[0], dec_nf[3]))  # 3
        self.dec.append(conv_block(dim, dec_nf[3] + dim, dec_nf[4]))  # 4

        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, x):
        """
        Pass input x through the UNet forward once
            :param x: concatenated fixed and moving image
        """
        # Get encoder activations
        x_enc = [x]
        for l in self.enc:
            x_enc.append(l(x_enc[-1]))

        # Three conv + upsample + concatenate series
        y = x_enc[-1]
        for i in range(3):
            y = self.dec[i](y)
            y = self.upsample(y)
            y = torch.cat([y, x_enc[-(i + 2)]], dim=1)

        # One convs at full_size/2 res
        y = self.dec[3](y)

        # Upsample to full res, concatenate and conv
        y = self.upsample(y)
        y = torch.cat([y, x_enc[0]], dim=1)
        y = self.dec[4](y)

        return y

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1,
                 linear=False):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.linear = linear
        self.sr_ratio = sr_ratio
        if not linear:
            if sr_ratio > 1:
                self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
                self.norm = nn.LayerNorm(dim)
        else:
            self.pool = nn.AdaptiveAvgPool2d(7)
            self.sr = nn.Conv2d(dim, dim, kernel_size=1, stride=1)
            self.norm = nn.LayerNorm(dim)
            self.act = nn.GELU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if not self.linear:
            if self.sr_ratio > 1:
                x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
                x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
                x_ = self.norm(x_)
                kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            else:
                kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(self.pool(x_)).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            x_ = self.act(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

def DropPath(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, xHW):
        x = xHW[0]
        H = xHW[1]
        W = xHW[2]
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        S = x.shape
        x = x.transpose(1, 2).view(-1, S[1], H, W)

        return x

class Mlp2(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act1 = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.act2 = act_layer()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.dwconv(x, H, W)
        x = self.act2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class OverlapPatchEmbedBlock(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, dim, num_heads, img_size=128, patch_size=7, stride=4, in_chans=3, embed_dim=768, flatten=True,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1, linear=False):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.flatten = flatten
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio, linear=linear)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp2(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        _, C, H, W = x.shape
        if self.flatten:
            # BCHW -> BNC
            x = x.flatten(2).transpose(1, 2)
        x = self.act(x)
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        x = x.transpose(1, 2).view(-1, C, H, W)

        return x

class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=128, patch_size=7, stride=4, in_chans=3, embed_dim=768, flatten=True):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // stride, img_size[1] // stride
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.flatten = flatten

        self.act = nn.GELU()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        if self.flatten:
            # BCHW -> BNC
            x = x.flatten(2).transpose(1, 2)
        x = self.act(x)

        return x, H, W

class conv_block(nn.Module):
    """
    [conv_block] represents a single convolution block in the Unet which
    is a convolution based on the size of the input channel and output
    channels and then preforms a Leaky Relu with parameter 0.2.
    """

    def __init__(self, dim, in_channels, out_channels, stride=1):
        """
        Instiatiate the conv block
            :param dim: number of dimensions of the input
            :param in_channels: number of input channels
            :param out_channels: number of output channels
            :param stride: stride of the convolution
        """
        super(conv_block, self).__init__()

        conv_fn = getattr(nn, "Conv{0}d".format(dim))

        if stride == 1:
            ksize = 3
        elif stride == 2:
            ksize = 4
        else:
            raise Exception('stride must be 1 or 2')

        self.main = conv_fn(in_channels, out_channels, ksize, stride, 1)
        self.activation = nn.LeakyReLU(0.2)

    def forward(self, x):
        """
        Pass the input through the conv_block
        """
        out = self.main(x)
        out = self.activation(out)
        return out


class SpatialTransformer(nn.Module):
    """
    [SpatialTransformer] represesents a spatial transformation block
    that uses the output of the deformation flow to preform an grid_sample
    https://pytorch.org/docs/stable/nn.functional.html#grid-sample
    """

    def __init__(self, size, mode='bilinear'):
        """
        Instiatiate the block
            :param size: size of input to the spatial transformer block
            :param mode: method of interpolation for grid_sampler
        """
        super(SpatialTransformer, self).__init__()

        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)

        self.mode = mode

    def forward(self, src, flow):
        """
        Push the src and flow through the spatial transform block
            :param src: the original moving image
            :param flow: the output from the U-Net
        """
        new_locs = self.grid + flow

        shape = flow.shape[2:]

        # Need to normalize grid values to [-1, 1] for resampler
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

        if len(shape) == 2:
            new_locs = new_locs.permute(0, 2, 3, 1)
            new_locs = new_locs[..., [1, 0]]
        elif len(shape) == 3:
            new_locs = new_locs.permute(0, 2, 3, 4, 1)
            new_locs = new_locs[..., [2, 1, 0]]

        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)


class DiffeomorphicTransform(nn.Module):
    def __init__(self, size, mode='bilinear', time_step=7):
        super(DiffeomorphicTransform, self).__init__()
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in size]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.FloatTensor)
        self.register_buffer('grid', grid)
        self.mode = mode
        self.time_step = time_step

    def forward(self, velocity):
        flow = velocity / (2.0 ** self.time_step)
        # 1.0 flow
        for _ in range(self.time_step):
            new_locs = self.grid + flow
            shape = flow.shape[2:]
            # Need to normalize grid values to [-1, 1] for resampler
            for i in range(len(shape)):
                new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)

            if len(shape) == 2:
                new_locs = new_locs.permute(0, 2, 3, 1)
                new_locs = new_locs[..., [1, 0]]
            elif len(shape) == 3:
                new_locs = new_locs.permute(0, 2, 3, 4, 1)
                new_locs = new_locs[..., [2, 1, 0]]
            flow = flow + nnf.grid_sample(flow, new_locs, align_corners=True, mode=self.mode)
        return flow


class Lagrangian_flow_refinement(nn.Module):
    """
    [Lagrangian_flow_refinement] is a class representing the computation of Lagrangian flow (v12, v13, v14, ...) from inter frame
    (INF) flow filed (u2'2, u3'3, u4'4, ...) and the coarse Lagrangian flow (v12', v13', v14',...)
    v12 = v12' + u2'2 o v12' ('o' is a warping)
    v13 = v13' + u3'3 o v13'
    v14 = v14' + u4'4 o v14'
    ...
    """

    def __init__(self, vol_size):
        """
        Instiatiate Lagrangian_flow layer
            :param vol_size: volume size of the atlas
        """
        super(Lagrangian_flow_refinement, self).__init__()

        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, inf_flow, lag_flow):
        """
        Pass input x through forward once
            :param inf_flow: inter frame motion field
            :param lag_flow: coarse Lagrangian motion field
        """
        shape = inf_flow.shape
        seq_len = shape[0]
        refined_lag_flow = lag_flow.clone()
        for k in range (0, seq_len):
            src = inf_flow[k, ::]
            sum_flow = lag_flow[k:k+1, ::]
            src_x = src[0, ::]
            src_x = src_x.unsqueeze(0)
            src_x = src_x.unsqueeze(0)
            src_y = src[1, ::]
            src_y = src_y.unsqueeze(0)
            src_y = src_y.unsqueeze(0)
            lag_flow_x = self.spatial_transform(src_x, sum_flow)
            lag_flow_y = self.spatial_transform(src_y, sum_flow)
            refined_lag_flow[k, ::] = sum_flow + torch.cat((lag_flow_x, lag_flow_y), dim=1)

        return refined_lag_flow


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=False, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class Mlp(nn.Module):
    def __init__(self, in_features):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(in_features, 64, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(64, 6, bias=False),
            nn.Tanh()
        )

    def forward(self, x):
        affine_params = self.head(x.mean(dim=1))
        return affine_params


class AffineCOMTransform(nn.Module):
    def __init__(self, use_com=True):
        super(AffineCOMTransform, self).__init__()

        self.translation_m = None
        self.rotation_x = None
        self.rotation_y = None
        self.rotation_z = None
        self.rotation_m = None
        self.shearing_m = None
        self.scaling_m = None

        self.id = torch.zeros((1, 2, 3)).cuda()
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1

        self.use_com = use_com

    def restore_flow_from_grid(self, flow):
        flow = flow[..., [1, 0]]  # attention!!!
        flow = flow.permute(0, 3, 1, 2)
        shape = flow.shape[2:]
        # Create sampling grid
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)  # y, x, z
        grid = grid.to(flow.device)
        grid = torch.unsqueeze(grid, 0)  # add batch
        grid = grid.type(torch.cuda.FloatTensor)
        # Need to restore grid values from [-1, 1] to [0, shape-1]
        for i in range(len(shape)):
            flow[:, i, ...] = 0.5 * (flow[:, i, ...] + 1) * (shape[i] - 1)

        return flow - grid  # B*2*H*W

    def forward(self, x, affine_para, test=False):
        # Matrix that register x to its center of mass
        id_grid = F.affine_grid(self.id, x.shape, align_corners=True)

        to_center_matrix = torch.eye(3).cuda()
        reversed_to_center_matrix = torch.eye(3).cuda()
        if self.use_com:
            x_sum = torch.sum(x)
            center_mass_x = torch.sum(x.permute(0, 2, 3, 1)[..., 0] * id_grid[..., 0]) / x_sum
            center_mass_y = torch.sum(x.permute(0, 2, 3, 1)[..., 0] * id_grid[..., 1]) / x_sum

            to_center_matrix[0, 2] = center_mass_x
            to_center_matrix[1, 2] = center_mass_y

            reversed_to_center_matrix[0, 2] = -center_mass_x
            reversed_to_center_matrix[1, 2] = -center_mass_y

        self.translation_m = torch.eye(3).cuda()
        self.rotation_x = torch.eye(3).cuda()
        self.rotation_y = torch.eye(3).cuda()
        self.rotation_z = torch.eye(3).cuda()
        self.rotation_m = torch.eye(3).cuda()
        self.shearing_m = torch.eye(3).cuda()
        self.scaling_m = torch.eye(3).cuda()

        trans_xyz = affine_para[0, 0:2]
        rotate_xyz = affine_para[0, 2:3] * math.pi
        shearing_xyz = affine_para[0, 3:4] * math.pi
        scaling_xyz = 1 + (affine_para[0, 4:6] * 0.5)

        self.translation_m[0, 2] = trans_xyz[0]
        self.translation_m[1, 2] = trans_xyz[1]

        self.scaling_m[0, 0] = scaling_xyz[0]
        self.scaling_m[1, 1] = scaling_xyz[1]

        self.rotation_z[0, 0] = torch.cos(rotate_xyz[0])
        self.rotation_z[0, 1] = -torch.sin(rotate_xyz[0])
        self.rotation_z[1, 0] = torch.sin(rotate_xyz[0])
        self.rotation_z[1, 1] = torch.cos(rotate_xyz[0])

        self.shearing_m[0, 1] = shearing_xyz[0]


        output_affine_m = torch.mm(to_center_matrix, torch.mm(self.shearing_m, torch.mm(self.scaling_m,
                                                                                        torch.mm(self.rotation_z,
                                                                                                 torch.mm(reversed_to_center_matrix,
                                                                                                     self.translation_m)))))
        grid = F.affine_grid(output_affine_m[0:2].unsqueeze(0), x.shape, align_corners=True)
        transformed_x = F.grid_sample(x, grid, mode='bilinear', align_corners=True)
        if test:
            affine_flow = self.restore_flow_from_grid(grid)
            return transformed_x, affine_flow

        return transformed_x, output_affine_m[0:2].unsqueeze(0)


class Affine_diffeo_reg_net(nn.Module):
    """
    [Affine_diffeo_reg_net] is a class representing the architecture for cascaded affine and non-rigid deformation
    estimation. The backbone is a TransUNet model.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, int_steps=7):
        """
        Instiatiate lagrangian_motion_estimate_net model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
            :param int_steps: the number of integration steps
        """
        super(Affine_diffeo_reg_net, self).__init__()

        dim = len(vol_size)

        self.vit_unet_model = vit_unet_core(dim, enc_nf, dec_nf)
        self.affine_mlp = Mlp(in_features=dec_nf[-1])
        self.flow_conv = conv_block(dim, dec_nf[-1], dim)

        self.diffeomorph_transform = DiffeomorphicTransform(size=vol_size, mode='bilinear', time_step=int_steps)
        self.spatial_transform = SpatialTransformer(vol_size)
        self.lag_mc = Lagrangian_flow_refinement(vol_size)
        self.affine_transform = AffineCOMTransform()


    def forward(self, x, y, netG, detach=False):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        # x: cine, y: tagging
        fake_y, corr_warped_cine, corr_warped_cine_cycle, _, corr_flow = netG(y, x)

        if detach:
            fake_y = fake_y.detach()

        affine_only = False
        deform_only = False
        if affine_only:
            # affine flows
            cat_input = torch.cat((x, fake_y), 1)
            z = self.vit_unet_model(cat_input)
            _, C, H, W = z.shape  # 1*32*12*12

            # BCHW -> BNC
            z = z.flatten(2).transpose(1, 2)

            affine_params = self.affine_mlp(z)
            warpped_x_1, affine_flow = self.affine_transform(x, affine_params, test=True)
            warpped_x_2 = None
            deform_flow = None
            warpped_x_12 = warpped_x_1
            final_flow = affine_flow
        elif deform_only:
            cat_input = torch.cat((x, fake_y), 1)
            z = self.vit_unet_model(cat_input)

            z = self.flow_conv(z)
            deform_flow = self.diffeomorph_transform(z)
            warpped_x_2 = self.spatial_transform(x, deform_flow)
            warpped_x_1 = None
            affine_flow = None
            warpped_x_12 = warpped_x_2
            final_flow = deform_flow
        else:
            # affine flows
            cat_input = torch.cat((x, fake_y), 1)
            z = self.vit_unet_model(cat_input)
            _, C, H, W = z.shape # 1*32*12*12

            # BCHW -> BNC
            z = z.flatten(2).transpose(1, 2)

            affine_params = self.affine_mlp(z)
            warpped_x_1, affine_flow = self.affine_transform(x, affine_params, test=True)

            # diffeomorphic nonrigid flows
            cat_input = torch.cat((warpped_x_1, fake_y), 1)
            z = self.vit_unet_model(cat_input)

            z = self.flow_conv(z)
            deform_flow = self.diffeomorph_transform(z)
            warpped_x_2 = self.spatial_transform(warpped_x_1, deform_flow)
            final_flow = self.lag_mc(affine_flow, deform_flow)
            warpped_x_12 = self.spatial_transform(x, final_flow)

        return warpped_x_1, affine_flow, fake_y, corr_warped_cine, corr_warped_cine_cycle, corr_flow, warpped_x_2, deform_flow, warpped_x_12, final_flow


class Affine_diffeo_reg_net2(nn.Module):
    """
    [Affine_diffeo_reg_net2] is a class representing the architecture for cascaded affine and non-rigid deformation
    estimation. The backbone is a pure CNN-based UNet model.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, int_steps=7):
        """
        Instiatiate lagrangian_motion_estimate_net model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
            :param int_steps: the number of integration steps
        """
        super(Affine_diffeo_reg_net2, self).__init__()

        dim = len(vol_size)

        self.vit_unet_model = unet_core2(dim, enc_nf, dec_nf)
        self.affine_mlp = Mlp(in_features=dec_nf[-1])
        self.flow_conv = conv_block(dim, dec_nf[-1], dim)

        self.diffeomorph_transform = DiffeomorphicTransform(size=vol_size, mode='bilinear', time_step=int_steps)
        self.spatial_transform = SpatialTransformer(vol_size)
        self.lag_mc = Lagrangian_flow_refinement(vol_size)
        self.affine_transform = AffineCOMTransform()


    def forward(self, x, y, netG, detach=False):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        # x: cine, y: tagging
        fake_y, corr_warped_cine, corr_warped_cine_cycle, _, corr_flow = netG(y, x)
        if detach:
            fake_y = fake_y.detach()

        affine_only = False
        deform_only = False
        if affine_only:
            # affine flows
            cat_input = torch.cat((x, fake_y), 1)
            z = self.vit_unet_model(cat_input)
            _, C, H, W = z.shape  # 1*32*12*12

            # BCHW -> BNC
            z = z.flatten(2).transpose(1, 2)

            affine_params = self.affine_mlp(z)
            warpped_x_1, affine_flow = self.affine_transform(x, affine_params, test=True)
            warpped_x_2 = None
            deform_flow = None
            warpped_x_12 = warpped_x_1
            final_flow = affine_flow
        elif deform_only:
            cat_input = torch.cat((x, fake_y), 1)
            z = self.vit_unet_model(cat_input)

            z = self.flow_conv(z)
            deform_flow = self.diffeomorph_transform(z)
            warpped_x_2 = self.spatial_transform(x, deform_flow)
            warpped_x_1 = None
            affine_flow = None
            warpped_x_12 = warpped_x_2
            final_flow = deform_flow
        else:
            # affine flows
            cat_input = torch.cat((x, fake_y), 1)
            z = self.vit_unet_model(cat_input)
            _, C, H, W = z.shape # 1*32*12*12

            # BCHW -> BNC
            z = z.flatten(2).transpose(1, 2)

            affine_params = self.affine_mlp(z)
            warpped_x_1, affine_flow = self.affine_transform(x, affine_params, test=True)

            # diffeomorphic nonrigid flows
            cat_input = torch.cat((warpped_x_1, fake_y), 1)
            z = self.vit_unet_model(cat_input)

            z = self.flow_conv(z)
            deform_flow = self.diffeomorph_transform(z)
            warpped_x_2 = self.spatial_transform(warpped_x_1, deform_flow)
            final_flow = self.lag_mc(affine_flow, deform_flow)
            warpped_x_12 = self.spatial_transform(x, final_flow)

        return warpped_x_1, affine_flow, fake_y, corr_warped_cine, corr_warped_cine_cycle, corr_flow, warpped_x_2, deform_flow, warpped_x_12, final_flow

class Lagrangian_flow(nn.Module):
    """
    [Lagrangian_flow] is a class representing the computation of Lagrangian flow (v12, v13, v14, ...) from inter frame
    (INF) flow filed (u12, u23, u34, ...). The backward Lagrangian flows are computed as:
    v12 = u12
    v13 = v12 + u23 o v12 ('o' is a warping)
    v14 = v13 + u34 o v13
    ...
    """

    def __init__(self, vol_size):
        """
        Instiatiate Lagrangian_flow layer
            :param vol_size: volume size of the atlas
        """
        super(Lagrangian_flow, self).__init__()

        self.spatial_transform = SpatialTransformer(vol_size)



    def forward(self, inf_flow, forward_flow=True):
        """
        Pass input x through forward once
            :param inf_flow: inter frame motion field
        """
        shape = inf_flow.shape
        seq_len = shape[0]
        lag_flow = torch.zeros(shape, device=inf_flow.device)
        lag_flow[0, ::] = inf_flow[0, ::]
        for k in range(1, seq_len):
            if forward_flow:
                src = lag_flow[k-1, ::].clone()
                sum_flow = inf_flow[k:k+1, ::]
            else:
                src = inf_flow[k, ::]
                sum_flow = lag_flow[k - 1:k, ::]
            src_x = src[0, ::]
            src_x = src_x.unsqueeze(0)
            src_x = src_x.unsqueeze(0)
            src_y = src[1, ::]
            src_y = src_y.unsqueeze(0)
            src_y = src_y.unsqueeze(0)
            lag_flow_x = self.spatial_transform(src_x, sum_flow)
            lag_flow_y = self.spatial_transform(src_y, sum_flow)
            lag_flow[k, ::] = sum_flow + torch.cat((lag_flow_x, lag_flow_y), dim=1)

        return lag_flow

class Lagrangian_motion_estimate_net(nn.Module):
    """
    [lagrangian_motion_estimate_net] is a class representing the architecture for Lagrangian motion estimation on a time
     sequence which is based on a UNet model with a full sequence of lagrangian motion constraints. You may need to
     modify this code (e.g., number of layers) to suit your project needs.
    """

    def __init__(self, vol_size, enc_nf, dec_nf, test_phase=False, full_size=True, int_steps=7):
        """
        Instiatiate lagrangian_motion_estimate_net model
            :param vol_size: volume size of the atlas
            :param enc_nf: the number of features maps for encoding stages
            :param dec_nf: the number of features maps for decoding stages
            :param full_size: boolean value full amount of decoding layers
            :param int_steps: the number of integration steps
        """
        super(Lagrangian_motion_estimate_net, self).__init__()

        dim = len(vol_size)

        self.unet_model = unet_core(dim, enc_nf, dec_nf, full_size)

        self.diffeomorph_transform = DiffeomorphicTransform(size=vol_size, mode='bilinear',time_step=int_steps)
        self.spatial_transform = SpatialTransformer(vol_size)
        self.lag_flow = Lagrangian_flow(vol_size)
        self.lag_regular = True
        self.test_phase = test_phase

    def forward(self, src, tgt):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param tgt: fixed image that we want to shift to
        """
        x = torch.cat([src, tgt], dim=1)
        z = self.unet_model(x)
        # del x

        # bi-directional INF flows
        inf_flow = self.diffeomorph_transform(z)
        neg_inf_flow = self.diffeomorph_transform(-z)
        # del z

        # image warping
        y_src = self.spatial_transform(src, inf_flow)
        y_tgt = self.spatial_transform(tgt, neg_inf_flow)

        if self.lag_regular:
            # Lagrangian flow
            lag_flow = self.lag_flow(inf_flow)
            # Warp the reference frame by the Lagrangian flow
            src_0 = src[0, ::]
            shape = src.shape  # seq_length (batch_size), channel, height, width
            seq_length = shape[0]
            src_re = src_0.repeat(seq_length, 1, 1, 1)  # repeat the 1st frame to match other frames contained in a sequence
            src_re = src_re.contiguous()
            lag_y_src = self.spatial_transform(src_re, lag_flow)
            return y_src, y_tgt, lag_y_src, inf_flow, neg_inf_flow, lag_flow
        else:
            return y_src, y_tgt, inf_flow, neg_inf_flow


class miccai2018_net_cc_san_grid_warp(nn.Module):
    """
    [miccai2018_net_cc_san_grid_warp] is a class for image warping during inference.
    """

    def __init__(self, vol_size):
        """
        Instiatiate 2018 model
            :param vol_size: volume size of the atlas
        """
        super(miccai2018_net_cc_san_grid_warp, self).__init__()

        self.spatial_transform = SpatialTransformer(vol_size)

    def forward(self, src, flow):
        """
        Pass input x through forward once
            :param src: moving image that we want to shift
            :param flow: motion field used to warp src image
        """
        y_src = self.spatial_transform(src, flow)

        return y_src


class Diffeo_reg_loss(torch.nn.Module):
    """
    Diffeo_reg_loss
    """

    def __init__(self):
        super(Diffeo_reg_loss, self).__init__()

    def l1_loss(self, y_true, y_pred):
        """ reconstruction loss """
        # return  torch.mean(torch.abs((y_true - y_pred)))
        L1_loss = torch.nn.L1Loss()
        return L1_loss(y_true, y_pred)

    def l2_loss(self, y_true, y_pred):
        """ reconstruction loss """
        return torch.mean((y_true - y_pred) ** 2)

    def weighted_loss(self, warped_grid, fixed_img):
        """ weighted loss """
        s = warped_grid.shape
        one_matrix = torch.ones(s).cuda()
        reversed_grid = one_matrix - warped_grid
        # reversed_grid = reversed_grid.cuda()
        return torch.mean(reversed_grid * fixed_img)

    def gradient_loss(self, s, penalty='l2'):
        # s is the deformation_matrix of shape (seq_length, channels=2, height, width)
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])

        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0
    
    def bendingenergy_loss(self, s, penalty='l2'):
        # code from: https://github.com/uncbiag/DeepAtlas/blob/master/lib/loss.py
        # s is the deformation_matrix of shape (seq_length, channels=2, height, width)
        # according to f''(x) = [f(x+h) + f(x-f) - 2f(x)]/h^2
        # f_{x, y}(x, y) = [df(x+h, y+k) + df(x-h, y-k) - df(x+h, y-k) - df(x-h, y+k)] / 2hk
        ddx = torch.abs(s[:, :, 2:, 1:-1] + s[:, :, :-2, 1:-1] - 2*s[:, :, 1:-1, 1:-1]).view(s.shape[0], s.shape[1], -1)
        ddy = torch.abs(s[:, :, 1:-1, 2:] + s[:, :, 1:-1, :-2] - 2*s[:, :, 1:-1, 1:-1]).view(s.shape[0], s.shape[1], -1) 
        dxdy = torch.abs(s[:, :, 2:, 2:] + s[:, :, :-2, :-2] - s[:, :, 2:, :-2] - s[:, :, :-2, 2:]).view(s.shape[0], s.shape[1], -1) 
        

        if (penalty == 'l2'):
            ddx = (ddx ** 2).mean(2)
            ddy = (ddy ** 2).mean(2)
            dxdy = (dxdy ** 2).mean(2)
        d = (ddx.mean() + ddy.mean() + 2*dxdy.mean()) / 4.0
        
        return d


class NCC(torch.nn.Module):
    """
    local (over window) normalized cross correlation
    """

    def __init__(self, win=9, eps=1e-3):
        super(NCC, self).__init__()
        self.win_raw = win
        self.eps = eps
        self.win = win

    def forward(self, I, J):
        ndims = 2
        win_size = self.win_raw
        self.win = [self.win_raw] * ndims

        weight_win_size = self.win_raw
        weight = torch.ones((1, 1, weight_win_size, weight_win_size), device=I.device, requires_grad=False)
        # prepare conv kernel
        conv_fn = getattr(F, 'conv%dd' % ndims)
        # conv_fn = F.conv3d

        # compute CC squares
        I2 = I * I
        J2 = J * J
        IJ = I * J

        # compute filters
        # compute local sums via convolution
        I_sum = conv_fn(I, weight, padding=int(win_size / 2))
        J_sum = conv_fn(J, weight, padding=int(win_size / 2))
        I2_sum = conv_fn(I2, weight, padding=int(win_size / 2))
        J2_sum = conv_fn(J2, weight, padding=int(win_size / 2))
        IJ_sum = conv_fn(IJ, weight, padding=int(win_size / 2))

        # compute cross correlation
        # win_size = np.prod(self.win)
        win_size = torch.from_numpy(np.array([np.prod(self.win)])).float()
        win_size = win_size.cuda()
        u_I = I_sum / win_size
        u_J = J_sum / win_size

        cross = IJ_sum - u_J * I_sum - u_I * J_sum + u_I * u_J * win_size
        I_var = I2_sum - 2 * u_I * I_sum + u_I * u_I * win_size
        J_var = J2_sum - 2 * u_J * J_sum + u_J * u_J * win_size

        cc0 = cross * cross / (I_var * J_var + self.eps)
        cc = torch.clamp(cc0, 0.001, 0.999)

        # return negative cc.
        return -1.0 * torch.mean(cc)
