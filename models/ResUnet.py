################ This code is modified and adapted for task of denoising diffusion model : https://github.com/ChienWong/ResUnet-a
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

################################################################### Helper function from Lucid drains diffusion github
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
    # see https://pytorch.org/docs/0.4.0/_modules/torchvision/models/resnet.html
    def __init__(self, inplanes, planes,kernel,dilations,emb_channels,stride=1):
        super(BasicBlock, self).__init__()
        self.u=nn.ModuleList([])
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                inplanes,
            )
        )
        for d in dilations:
            layer = nn.Sequential(
                nn.BatchNorm2d(inplanes),
                nn.ReLU(),
                nn.Conv2d(inplanes,planes,kernel_size=kernel,dilation=d,stride=stride,padding='same'),
                nn.BatchNorm2d(planes),
                nn.ReLU(),              
                nn.Conv2d(inplanes,planes,kernel_size=kernel,dilation=d,stride=stride,padding='same')
            )
            self.u.append(layer)

    def forward(self, x,emb):
        emb_out =  self.emb_layers(emb).type(x.dtype)
        while len(emb_out.shape) < len(x.shape):
            emb_out = emb_out[..., None]
        x += emb_out
        # x = x.to('cuda')
        t =torch.zeros_like(x).to('cuda')
        if len(self.u)>1:
            for i in self.u:
                t+=i(x)
        else:
            t=self.u[0](x)
        return t
class PSPpooling(nn.Module):
    """Some Information about PSPpooling"""
    def __init__(self,filter,in_ch):
        super(PSPpooling, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        self.pool2 = nn.MaxPool2d(kernel_size=4)
        self.conv1 = nn.Conv2d(in_ch,int(filter/2),kernel_size=1)
        self.conv2 = nn.Conv2d(in_ch,int(filter/2),kernel_size=1)
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=4)
        self.conv0 = nn.Conv2d(filter+in_ch,filter,kernel_size=1)
    def forward(self, x):
        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1 = self.up1(x1)
        x2 = self.up2(x2)
        # print(x.shape) 
        # print(x1.shape)
        # print(x2.shape)
        x3 = torch.cat((x1,x2,x),dim=1).to('cuda')
        x3 = self.conv0(x3)

        return x3
class Combine(nn.Module):
    """Some Information about Combine"""
    def __init__(self,in_ch,filter):
        super (Combine, self).__init__()
        self.reluu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch,filter,kernel_size=1)
    def forward(self, inp1,inp2):
        inp1 = self.reluu(inp1)
        x = torch.cat((inp1,inp2),dim=1).to('cuda')
        x = self.conv1(x)
        return x


        return x
class ResUnet(nn.Module):
    """Some Information about ResUnet"""
    def __init__(
            self,
        dim,
        in_ch = 1,
        out_ch = None,
        dim_mults=(1, 2, 4, 8),
        channels = 1,
        encoder_relu=False,
        decoder_relu=True,
        self_condition = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
        ):
        super().__init__()
        self.channels = channels
        self.init_dim = 1
        self.out_dim = 1
        self.self_condition = self_condition
        # determine dimensions
        time_dim = dim * 4
        # self.channels = channels
        # self.initial_block = InitialBlock(in_ch, 16, relu=encoder_relu)

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(
            sinu_pos_emb,
            nn.Linear(fourier_dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim)
        )
        self.inp = nn.Conv2d(in_ch,32,kernel_size=1,stride=1,dilation=1)
        self.c1  = BasicBlock(32,32,kernel=3,dilations=[1,3],stride=1,emb_channels=time_dim)
        self.a1  = Residual(PreNorm(32, LinearAttention(32)))
        self.s1  = nn.Conv2d(32,64,kernel_size=1,stride=2)
        self.c2  = BasicBlock(64,64,kernel=3,dilations=[1,3],stride=1,emb_channels=time_dim)
        self.a2  = Residual(PreNorm(64, LinearAttention(64)))
        self.s2  = nn.Conv2d(64,128,kernel_size=1,stride=2)
        self.c3  = BasicBlock(128,128,kernel=3,dilations=[1,3],stride=1,emb_channels=time_dim)
        self.a3  = Residual(PreNorm(128, LinearAttention(128)))
        self.po  = PSPpooling(16,128)
        self.up1 = nn.Upsample(scale_factor=2)
        self.up2 = nn.Upsample(scale_factor=2)
        self.co1 = Combine(80,32)
        self.c4  = BasicBlock(32,32,kernel=3,dilations=[1,3],stride=1,emb_channels=time_dim)
        self.a4  = Residual(PreNorm(32, LinearAttention(32)))
        self.co2 = Combine(64,32)
        self.c5  = BasicBlock(32,32,kernel=3,dilations=[1,3],stride=1,emb_channels=time_dim)
        self.a5  = Residual(PreNorm(32, LinearAttention(32)))
        self.conv0 = nn.Conv2d(32,in_ch,kernel_size=1)

    def forward(self, x,t, x_self_cond = None):

        t1 = self.time_mlp(t)

        x = self.inp(x)
        d1 = self.c1(x,t1)
        x = self.a1(d1)
        x = self.s1(x)
        d2 = self.c2(x,t1)
        x  = self.a2(d2)
        x = self.s2(x)
        d3 = self.c3(x,t1)
        x  = self.a3(d3)
        x = self.po(x)
        x = self.up1(x)
        x = self.co1(x,d2)
        x = self.c4(x,t1)
        x  = self.a4(x)
        x = self.up2(x)
        # print("x shape",x.shape)
        # print("d1 shape",d1.shape)
        x = self.co2(x,d1)
        x = self.c5(x,t1)
        x  = self.a5(x)
        x = self.conv0(x)

        return x