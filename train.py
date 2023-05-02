from denoise import GaussianDiffusion, Trainer
from models.Enet import ENet
from models.LinkNet import LinkNetBaseMNIST
from models.ResUnet import ResUnet
from models.Segnet import SegNetBase
from models.Xnet import XNetBase
from models.Unet import Unet

# model = ENet(
#     dim = 64,
#     ch=1,
#     # in_ch=1,
#     dim_mults = (1, 2 ,4)
# )
# model = LinkNetBaseMNIST(
#     dim = 64,
#     dim_mults = (1, 2, 4, 8))
# model = ResUnet(
#     dim = 64,
#     channels=1,
#     # in_ch=1,
#     dim_mults = (1, 2 ,4)
# )
# model = SegNetBase(
#     dim = 64,
#     channels=1,
#     dim_mults = (1, 2, 4, 8)
# )
# model = XNetBase(
#     dim = 64,
#     channels=1,
#     dim_mults = (1, 2, 4, 8)
# )
model = Unet(
    dim = 64,
    channels=1,
    # in_ch=1,
    dim_mults = (1, 2 ,4)
)
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print("Total Model Parameters: ",pytorch_total_params)

########## Please set image size as 28 for Linknet and 32 for all other models.
diffusion = GaussianDiffusion(
    model,
    image_size = 32,
    # beta_schedule = 'cosine',
    timesteps = 1000,           # number of steps
    sampling_timesteps = 250,   # number of sampling timesteps (using ddim for faster inference [see citation for ddim paper])
    loss_type = 'l1'            # L1 or L2
)

trainer = Trainer(
    diffusion,
    './training_mnist',
    results_folder = './results/test',
    augment_horizontal_flip = False,
    save_and_sample_every = 1000,
    train_batch_size = 32,
    train_lr = 1e-4,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 2,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = False,                       # turn on mixed precision
    calculate_fid = True              # whether to calculate fid during training
)

# trainer.load()
trainer.train()