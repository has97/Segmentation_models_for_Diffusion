################ This code is modified and adapted for task of denoising diffusion model : https://github.com/e-lab/pytorch-linknet/
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

####################################### Helper function from Lucid drains diffusion github 
def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

def identity(t, *args, **kwargs):
    return t

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim, dim_out = None):
    return nn.Sequential(
        nn.Upsample(scale_factor = 2, mode = 'nearest'),
        nn.Conv2d(dim, default(dim_out, dim), 3, padding = 1)
    )

def Downsample(dim, dim_out = None):
    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (c p1 p2) h w', p1 = 2, p2 = 2),
        nn.Conv2d(dim * 4, default(dim_out, dim), 1)
    )
class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1', partial(torch.var, unbiased = False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride, self.padding, self.dilation, self.groups)

# sinusoidal positional embeds

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """

    def __init__(self, dim, is_random = False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim), requires_grad = not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim = -1)
        fouriered = torch.cat((x, fouriered), dim = -1)
        return fouriered
############################################################################

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups = 8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding = 1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift = None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim = None, groups = 8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_emb_dim, dim_out * 2)
        ) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups = groups)
        self.block2 = Block(dim_out, dim_out, groups = groups)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb = None):

        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, 'b c -> b c 1 1')
            scale_shift = time_emb.chunk(2, dim = 1)

        h = self.block1(x, scale_shift = scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)
class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim = 1, unbiased = False, keepdim = True)
        mean = torch.mean(x, dim = 1, keepdim = True)
        return (x - mean) * (var + eps).rsqrt() * self.g

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)
class LinearAttention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)

        self.to_out = nn.Sequential(
            nn.Conv2d(hidden_dim, dim, 1),
            LayerNorm(dim)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q.softmax(dim = -2)
        k = k.softmax(dim = -1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        o = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        o = rearrange(o, 'b h c (x y) -> b (h c) x y', h = self.heads, x = h, y = w)
        return self.to_out(o)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim = 1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h = self.heads), qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim = -1)
        o = einsum('b h i j, b h d j -> b h i d', attn, v)

        o = rearrange(o, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return self.to_out(o)
##################################################################################################################################################################    
class BasicBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=bias)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size, 1, padding, groups=groups, bias=bias)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.downsample = None
        if stride > 1:
            self.downsample = nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
                            nn.BatchNorm2d(out_planes),)

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

       #print("out1: ", type(out))
       #print("out2: ", out.shape)
       #print("out3: ", out[0].shape)
        return out


class Encoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, groups=1, bias=False, emb_channels=None):
        super(Encoder, self).__init__()
        self.block1 = BasicBlock(in_planes, out_planes, kernel_size, stride, padding, groups, bias)
        self.block2 = BasicBlock(out_planes, out_planes, kernel_size, 1, padding, groups, bias)
        self.res1 = Residual(PreNorm(out_planes, LinearAttention(out_planes)))
        
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                in_planes,
            )
        )
    def forward(self, x, emb):
        emb_out =  self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x += emb_out
        x = self.block1(x)
        x = self.block2(x)
       #print("x type", type(x))
       #print("res type", type(self.res1))
        x = self.res1(x)
        return x


class Decoder(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, output_padding=0, groups=1, bias=False, emb_channels = None):
        # TODO bias=True
        super(Decoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(in_planes, in_planes//4, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.tp_conv = nn.Sequential(nn.ConvTranspose2d(in_planes//4, in_planes//4, kernel_size, stride, padding, output_padding, bias=bias),
                                nn.BatchNorm2d(in_planes//4),
                                nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(in_planes//4, out_planes, 1, 1, 0, bias=bias),
                                nn.BatchNorm2d(out_planes),
                                nn.ReLU(inplace=True),)
        
        self.res1 = Residual(PreNorm(out_planes, LinearAttention(out_planes)))

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                in_planes,
            )
        )
    def forward(self, x, emb):
        emb_out =  self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x += emb_out
        
        x = self.conv1(x)
        x = self.tp_conv(x)
        x = self.conv2(x)
        x = self.res1(x)
        return x




class LinkNetBaseMNIST(nn.Module):
    """
    Generate model architecture
    """

    def __init__(        
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        n_classes = 1
        ):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        
        
        super(LinkNetBaseMNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, 7, 1, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.channels = channels
        self.init_dim = channels
        self.out_dim = channels
        self.self_condition = self_condition
        time_dim = dim * 4
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        ##print("sinu_pos_emb: ", sinu_pos_emb.shape)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.encoder1 = Encoder(64, 64, 3, 1, 1, emb_channels=time_dim)
        self.encoder2 = Encoder(64, 128, 3, 2, 1 , emb_channels=time_dim)
        self.encoder3 = Encoder(128, 256, 3, 2, 1, emb_channels=time_dim)
        self.encoder4 = Encoder(256, 512, 3, 2, 1, emb_channels=time_dim)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0  , emb_channels=time_dim)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1 , emb_channels=time_dim)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1, emb_channels=time_dim)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1, emb_channels=time_dim)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2, 0, 0),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 5, 1, 0),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 1, 1, 0)
        self.lsm = nn.LogSoftmax(dim=1)
        self.con2 = nn.ConvTranspose2d(128, 128, 2, 1, 0, 0, 1 , False, 1)
        self.con3 = nn.ConvTranspose2d(64, 64, 3, 1, 0, 0, 1 , False, 1)

    def forward(self, x, time, x_self_cond = None):
        # Initial block
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        
        t1 = self.time_mlp(time)
       #print("t1.shape: ", t1.shape)
        # Encoder blocks
        e1 = self.encoder1(x, t1)
       #print("e1 shape: ", e1.shape)
        e2 = self.encoder2(e1, t1)
       #print("e2 shape: ", e2.shape)
        e3 = self.encoder3(e2, t1)
       #print("e3 shape: ", e3.shape)
        e4 = self.encoder4(e3, t1)
       #print("e4 shape: ", e4.shape)

        # Decoder blocks
        #d4 = e3 + self.decoder4(e4)
        
       #print("e4 shape: ", e4.shape)
       #print("t1 shape: ", t1.shape)
        d41 = self.decoder4(e4, t1)
       #print("d41 shape: ", d41.shape)
       #print("e3 shape: ", e3.shape)
        d4 = e3 + d41
        
       #print("d4 shape: ", d4.shape)
        d31 = self.decoder3(d4, t1)
       #print("d31 shape: ", d31.shape)
       #print("e2 shape: ", e2.shape)
        e2 = self.con2(e2)
       #print("e21 shape: ", e2.shape)
        d3 = e2 + d31
        
        
       #print("d3 shape: ", d3.shape)
        d21 = self.decoder2(d3, t1)
       #print("d21 shape: ", d21.shape)
       #print("e1 shape: ", e1.shape)
        e1 = self.con3(e1)
       #print("e11 shape: ", e1.shape)
        d2 = e1 + d21
        
        
       #print("d2 shape: ", d2.shape)
        d11 = self.decoder1(d2, t1)
       #print("d11 shape: ", d11.shape)
       #print("x shape: ", x.shape)
        x = self.con3(x)
       #print("x1 shape: ", x.shape)
        d1 = x +  d11

        # Classifier
        y = self.tp_conv1(d1)
       #print("y1 shape: ", y.shape)
        y = self.conv2(y)
       #print("y2 shape: ", y.shape)
        y = self.tp_conv2(y)
       #print("y3 shape: ", y.shape)
        # y = self.lsm(y)

        return y
    
class LinkNetBaseSTL(nn.Module):
    """
    Generate model architecture
    """

    def __init__(        
        self,
        dim,
        init_dim = None,
        out_dim = None,
        dim_mults=(1, 2, 4, 8),
        channels = 3,
        self_condition = False,
        resnet_block_groups = 8,
        learned_variance = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16,
        n_classes = 3
        ):
        """
        Model initialization
        :param x_n: number of input neurons
        :type x_n: int
        """
        
        
        super(LinkNetBaseSTL, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 1, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.channels = channels
        self.self_condition = self_condition
        time_dim = dim * 4
        
        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        ##print("sinu_pos_emb: ", sinu_pos_emb.shape)
        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )

        self.encoder1 = Encoder(64, 64, 3, 1, 1, emb_channels=time_dim)
        self.encoder2 = Encoder(64, 128, 3, 2, 1 , emb_channels=time_dim)
        self.encoder3 = Encoder(128, 256, 3, 2, 1, emb_channels=time_dim)
        self.encoder4 = Encoder(256, 512, 3, 2, 1, emb_channels=time_dim)

        self.decoder1 = Decoder(64, 64, 3, 1, 1, 0  , emb_channels=time_dim)
        self.decoder2 = Decoder(128, 64, 3, 2, 1, 1 , emb_channels=time_dim)
        self.decoder3 = Decoder(256, 128, 3, 2, 1, 1, emb_channels=time_dim)
        self.decoder4 = Decoder(512, 256, 3, 2, 1, 1, emb_channels=time_dim)

        # Classifier
        self.tp_conv1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 2, 2, 0, 0),
                                      nn.BatchNorm2d(32),
                                      nn.ReLU(inplace=True),)
        self.conv2 = nn.Sequential(nn.Conv2d(32, 32, 1, 1, 0),
                                nn.BatchNorm2d(32),
                                nn.ReLU(inplace=True),)
        self.tp_conv2 = nn.ConvTranspose2d(32, n_classes, 1, 1, 0)
        self.lsm = nn.LogSoftmax(dim=1)
        self.con2 = nn.ConvTranspose2d(128, 128, 2, 1, 0, 0, 1 , False, 1)
        self.con3 = nn.ConvTranspose2d(64, 64, 3, 1, 0, 0, 1 , False, 1)

    def forward(self, x, time, x_self_cond = None):
        # Initial block
        
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        
        t1 = self.time_mlp(time)
        #print("t1.shape: ", t1.shape)
        # Encoder blocks
        e1 = self.encoder1(x, t1)
        #print("e1 shape: ", e1.shape)
        e2 = self.encoder2(e1, t1)
        #print("e2 shape: ", e2.shape)
        e3 = self.encoder3(e2, t1)
        #print("e3 shape: ", e3.shape)
        e4 = self.encoder4(e3, t1)
        #print("e4 shape: ", e4.shape)

        # Decoder blocks
        # d4 = e3 + self.decoder4(e4)

        #print("e4 shape: ", e4.shape)
        #print("t1 shape: ", t1.shape)
        d41 = self.decoder4(e4, t1)
        #print("d41 shape: ", d41.shape)
        #print("e3 shape: ", e3.shape)
        d4 = e3 + d41

        #print("d4 shape: ", d4.shape)
        d31 = self.decoder3(d4, t1)
        #print("d31 shape: ", d31.shape)
        #print("e2 shape: ", e2.shape)
        # e2 = self.con2(e2)
        # #print("e21 shape: ", e2.shape)
        d3 = e2 + d31


        #print("d3 shape: ", d3.shape)
        d21 = self.decoder2(d3, t1)
        #print("d21 shape: ", d21.shape)
        #print("e1 shape: ", e1.shape)
        # e1 = self.con3(e1)
        # #print("e11 shape: ", e1.shape)
        d2 = e1 + d21


        #print("d2 shape: ", d2.shape)
        d11 = self.decoder1(d2, t1)
        #print("d11 shape: ", d11.shape)
        #print("x shape: ", x.shape)
        # x = self.con3(x)
        # #print("x1 shape: ", x.shape)
        d1 = x +  d11

        # Classifier
        y = self.tp_conv1(d1)
        #print("y1 shape: ", y.shape)
        y = self.conv2(y)
        #print("y2 shape: ", y.shape)
        y = self.tp_conv2(y)
        #print("y3 shape: ", y.shape)
        # y = self.lsm(y)

        return y