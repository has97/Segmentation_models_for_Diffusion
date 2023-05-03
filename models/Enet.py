######## This code is modified and adapted for task of denoising diffusion model : https://github.com/davidtvs/PyTorch-ENet/tree/master/models
import torch
from torch import nn, einsum
import torch.nn.functional as F
import math
from functools import partial

from einops import rearrange, reduce
from einops.layers.torch import Rearrange

################################################################### Helper function from Lucid drains  github #################################
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
################################################################################################################################################# 
# Initial Block in Enet model
class InitialBlock(nn.Module):
    def __init__(self,in_ch,out_ch,bias=False,relu=True):
        super().__init__()
        # minus one because it is a grayscale image
        self.main1 = nn.Conv2d(in_ch,out_ch - 1,kernel_size=3,stride=2,padding=1,bias=bias)
        self.branch1 = nn.MaxPool2d(3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.PReLU()

    def forward(self, x):
        x1 = self.main1(x)
        x = self.branch1(x)

        x = torch.cat((x, x1), dim=1)
        x = self.bn(x)

        return self.act(x)
def linear(*args, **kwargs):
    return nn.Linear(*args, **kwargs)
class SiLU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)
# Bottleneck layer of Enet mode    
class Bottleneck(nn.Module):
    def __init__(self,
                 ch,
                 out_ch=None,
                 ratio=4,
                 kernel_size=3,
                 padding=0,
                 dilation=1,
                 asymmetric=False,
                 dropout_prob=0,
                 emb_channels = None,
                 bias=False,
                 relu=True):
        super().__init__()

        int_ch = ch // ratio # number of channels in the intermediate layers
        self.out_ch=out_ch
        act = nn.PReLU()
        self.conv1_1_1 = nn.Sequential(nn.Conv2d(ch,int_ch,1,1,bias=bias), nn.BatchNorm2d(int_ch), act)
        if asymmetric: # factorizing the convolution operation
            self.conv2 = nn.Sequential(
                nn.Conv2d(int_ch,int_ch,kernel_size=(kernel_size, 1),stride=1,padding=(padding, 0),dilation=dilation,bias=bias),
                nn.BatchNorm2d(int_ch), act,
                nn.Conv2d(int_ch,int_ch,kernel_size=(1, kernel_size),stride=1,padding=(0, padding),dilation=dilation,bias=bias), 
                nn.BatchNorm2d(int_ch), act)
        else: # no factorization of the intermediate convolution
            self.conv2 = nn.Sequential(
                nn.Conv2d(int_ch,int_ch,kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation,bias=bias), 
                nn.BatchNorm2d(int_ch), act)
        # time embbeding layer to the model for conditioning on the timestamps
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_channels,
                int_ch,
            )
        )

        # Now expanding to original dimension of channel using 1x1 convolution
        if out_ch is not None:
            self.conv3 = nn.Sequential(
                nn.Conv2d(int_ch,out_ch,1,1,bias=bias), 
                nn.BatchNorm2d(out_ch), act)
            self.conv4 = nn.Sequential(
                nn.Conv2d(ch,out_ch,kernel_size=kernel_size,stride=1,padding=padding,dilation=dilation,bias=bias), 
                nn.BatchNorm2d(out_ch), act)
        else:
            self.conv3 = nn.Sequential(
                nn.Conv2d(int_ch,ch,1,1,bias=bias), 
                nn.BatchNorm2d(ch), act)


        self.dropout = nn.Dropout2d(p=dropout_prob)

        self.out_act = act

    def forward(self, x, emb):
        x1     =  self.conv1_1_1(x)
        ####################### calculating the time embedding #####################################################33
        emb_out =  self.emb_layers(emb).type(x1.dtype)
        while len(emb_out.shape) < len(x1.shape):
            emb_out = emb_out[..., None]
        #############################################################################3
        x1 = self.conv2(x1)
        x1 += emb_out # adding the time stamp embedding for conditioning on it
        x1 = self.conv3(x1)
        x1 = self.dropout(x1)

        if self.out_ch is not None: # if the output channels is not specified
            o = self.conv4(x) + x1
        else:
            o = x + x1

        return self.out_act(o)

class DownsamplingBottleneck(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 ratio=4,
                pool_indices=False,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        self.pool_indices = pool_indices


        int_ch = in_ch // ratio

    
        act = nn.PReLU()

        # Maxpool done with return indices set as True for returning indices for upsampling
        self.pool1 = nn.MaxPool2d(2,stride=2,return_indices=pool_indices)

        #intermediate convolutions in downsampling layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch,int_ch,2,2,bias=bias), 
            nn.BatchNorm2d(int_ch), act)
        self.conv2 = nn.Sequential(
            nn.Conv2d(int_ch,int_ch,3,1,1,bias=bias), 
            nn.BatchNorm2d(int_ch), act)

        # Expanding to orginal channel dimension using the 1x1 convolution
        self.conv3 = nn.Sequential(
            nn.Conv2d(int_ch,out_ch,1,1,bias=bias), 
            nn.BatchNorm2d(out_ch), act)

        self.dropout = nn.Dropout2d(p=dropout_prob)
        self.out_act = act

    def forward(self, x):
        # Check if the pooling indices is to be returned
        if self.pool_indices:
            main, max_indices = self.pool1(x)
        else:
            main = self.pool1(x)

        # Contraction, Intermediate convolution and Expansion along other branch
        x1 = self.conv1(x)
        x1 = self.conv2(x1)
        x1 = self.conv3(x1)
        x1 = self.dropout(x1)

        # Now padding the main branch output with zeros to match the channel dimension.
        n, channel_branch, h, w = x1.size()
        channel_main = main.size()[1]
        padding = torch.zeros(n, channel_branch - channel_main, h, w)
        if main.is_cuda: # to make sure that the all are on the same devices
            padding = padding.cuda()
        main = torch.cat((main, padding), 1) # adding the zeros to the channel dimensions

        o = main + x1 # adding the output of main and other branch

        return self.out_act(o), max_indices
class UpsamplingBottleneck(nn.Module):

    def __init__(self,
                 in_ch,
                 out_ch,
                 ratio=4,
                 dropout_prob=0,
                 bias=False,
                 relu=True):
        super().__init__()

        int_ch = in_ch // ratio
        act = nn.PReLU()

        # The main branch were unpooling and 1x1 convolution for conversion to output channel
        self.mbranch_conv1 = nn.Sequential(
            nn.Conv2d(in_ch, out_ch,1, bias=bias),
            nn.BatchNorm2d(out_ch))
        
        self.mbranch_unpool1 = nn.MaxUnpool2d(kernel_size=2)

        # To decrease channels to int_ch using 1x1 convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_ch, int_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(int_ch), act)

        # Transposed convolution for upsampling in the other branch
        self.branch_conv1_t = nn.ConvTranspose2d(int_ch,int_ch,2,2,bias=bias)
        self.branch_norm = nn.BatchNorm2d(int_ch)
        self.branch_act = act

        # 1x1 expansion convolution to output channel dimension
        self.conv2 = nn.Sequential(
            nn.Conv2d(int_ch, out_ch, kernel_size=1, bias=bias),
            nn.BatchNorm2d(out_ch))

        self.dropout = nn.Dropout2d(p=dropout_prob)

        # output layer activation
        self.out_act = act

    def forward(self, x, max_indices, output_size):
        # The main branch where unpooling is done
        main = self.mbranch_conv1(x)
        main = self.mbranch_unpool1(main, max_indices, output_size=output_size)

        # side branch were the transpose convolution and other convolution operations are performed
        x1 = self.conv1(x)
        x1 = self.branch_conv1_t(x1, output_size=output_size)
        x1 = self.branch_norm(x1)
        x1 = self.branch_act(x1)
        x1 = self.conv2(x1)
        x1 = self.dropout(x1)

        # adding the output of the two branches together.
        o = main + x1

        return self.out_act(o)
#######################################################  The Enet model class ###########################################################################################    
class ENet(nn.Module):
    def __init__(
        self,
        dim,
        in_ch = 1,
        out_ch = None,
        dim_mults=(1, 2, 4, 8),
        ch = 1,
        encoder_relu=False,
        decoder_relu=True,
        self_condition = False,
        learned_sinusoidal_cond = False,
        random_fourier_features = False,
        learned_sinusoidal_dim = 16
    ):
        super().__init__()
        self.channels = ch
        self.init_dim = ch
        self.out_dim = ch
        self.self_condition = self_condition
        # determine dimensions
        time_dim = dim * 4
        # self.ch = ch
        self.initial_block = InitialBlock(in_ch, 16, relu=encoder_relu)

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
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        self.ups.append(nn.ModuleList([
                DownsamplingBottleneck(16,64,pool_indices=True,dropout_prob=0.01,relu=encoder_relu), #28->14
                Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(64, padding=1, dropout_prob=0.01, relu=encoder_relu,emb_channels=time_dim),
                Residual(PreNorm(64, LinearAttention(64))),
            ]))
        self.ups.append(nn.ModuleList([
                DownsamplingBottleneck(64,128,pool_indices=True,dropout_prob=0.1,relu=encoder_relu),#14->7
                Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128,kernel_size=5,padding=2,asymmetric=True,dropout_prob=0.1,relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128,kernel_size=5,asymmetric=True,padding=2,dropout_prob=0.1,relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Residual(PreNorm(128, LinearAttention(128))),
            ]))
        self.mids= nn.ModuleList([
                Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, dilation=2, padding=2, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128,kernel_size=5,padding=2,asymmetric=True,dropout_prob=0.1,relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, dilation=4, padding=4, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, padding=1, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, dilation=8, padding=8, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128,kernel_size=5,asymmetric=True,padding=2,dropout_prob=0.1,relu=encoder_relu,emb_channels=time_dim),
                Bottleneck(128, dilation=16, padding=16, dropout_prob=0.1, relu=encoder_relu,emb_channels=time_dim),
                Residual(PreNorm(128, LinearAttention(128))),
            ])
        self.downs.append(nn.ModuleList([
            UpsamplingBottleneck(128, 64, dropout_prob=0.1, relu=decoder_relu),
            Bottleneck(128,padding=1, dropout_prob=0.1, relu=decoder_relu,emb_channels=time_dim),
            Bottleneck(128, padding=1, dropout_prob=0.1, relu=decoder_relu,emb_channels=time_dim),
            Residual(PreNorm(128, LinearAttention(128))),      
        ]))
        self.downs.append(nn.ModuleList([
            UpsamplingBottleneck(128, 16, dropout_prob=0.1, relu=decoder_relu),
            Bottleneck(32, padding=1, dropout_prob=0.1, relu=decoder_relu,emb_channels=time_dim), 
            Residual(PreNorm(32, LinearAttention(32))),      
        ]))
        self.o = nn.ConvTranspose2d(32,in_ch,kernel_size=4,stride=2,padding=1,bias=False)
    def forward(self, x, time,x_self_cond = None):
        x = self.initial_block(x)
        t1 = self.time_mlp(time)
        maxs=[]
        sizes=[]
        max1=0
        us=[]
        for blocks in self.ups:
            t=0
            for b in blocks:
                if t==0:
                    sizes.append(x.size())
                    us.append(x)
                    x,max1 = b(x)
                    maxs.append(max1)
                elif t!=len(blocks)-1:
                    # print("Time shape ",time.size())
                    x = b(x,t1)
                else:
                    x=b(x)
                t+=1
        # print("X shape after up ",x.size())
        t=0
        for b in self.mids:
                if t!=len(self.mids)-1:
                    x=b(x,t1)
                else:
                    x=b(x)
                t+=1
        r = len(self.ups)-1
        # print("X shape after mid ",x.size())
        for blocks in self.downs:
            t=0
            for b in blocks:
                if t==0:
                    x = b(x,maxs[r],sizes[r])
                elif t!=len(blocks)-1:
                    if t==1:
                        x = b(torch.cat((x,us[r]),dim=1),t1)
                    else:
                        x = b(x,t1)
                else:
                    x=b(x)
                t+=1
            r-=1
        # print("X shape after up ",x.size())
        x = self.o(x)

        return x