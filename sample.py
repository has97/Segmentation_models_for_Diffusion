from denoise import GaussianDiffusion, Trainer
from models.Enet import ENet
from models.LinkNet import LinkNetBaseMNIST
from models.ResUnet import ResUnet
from models.Segnet import SegNetBase
from models.Xnet import XNetBase
from models.Unet import Unet
import os.path
from ema_pytorch import EMA

from torchvision import transforms as T, utils
import torch
import time

if __name__ == "__main__":
    # model = ENet(
    # dim = 64,
    # ch=1,
    # dim_mults = (1, 2, 4)
    # )
    model = LinkNetBaseMNIST(
    dim = 64,
    dim_mults = (1, 2, 4, 8))
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
    # model = Unet(
    # dim = 64,
    # channels=1,
    # # in_ch=1,
    # dim_mults = (1, 2 ,4)
    # )

    diffusion = GaussianDiffusion(
        model,
        image_size=32,
        timesteps=1000,  # number of steps
        loss_type='l1'  # L1 or L2
    ).cuda()

    data = torch.load("./model-9.pt",map_location=torch.device('cuda'))
    ema = EMA(diffusion, beta = 0.995, update_every = 10)
    ema.load_state_dict(data["ema"])
    ema.ema_model.eval()
    l=0
    # for k in range(6):
    start_time = time.time()
    images_tensor = ema.ema_model.sample(700)
    print("Time elapsed : ",time.time()-start_time)
    if(not os.path.exists("results_sample")):
            os.mkdir("results_sample")
    # images_tensor = ema.ema_model.sample(700)


    for i,image_tensors in enumerate(images_tensor):
            utils.save_image(image_tensors, "results_sample/ENet/samples"+str(i+l)+".png")
        # del images_tensor
        # l+=700
    #     print(image_tensor)
    #     image = transforms.ToPILImage(mode='L')(image_tensor)
    #     image.save(f"results_sample/sample_{i}.png")