import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np
import os
import glob
import cv2
from scipy import linalg

real_image_dir = './training_fmnist'
generated_image_dir = './results_sample'

img_size = (299, 299) # Because Inception Network is trained on this size

################################# Pre-process function ###################################################
def preprocess_image(image):
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB) # convertion required since the inception network is trained on RGB images
    image = cv2.resize(image, img_size, interpolation=cv2.INTER_LINEAR) # resize
    image = image.astype(np.float32) 
    image = ((image / 255.0) - 0.5) * 2.0
    return image
###########################################################################################################

def fid_score(real_images, gen_images):
    inception = models.inception_v3(pretrained=True, transform_input=False)
    inception.fc = nn.Identity() # to get the embbedings
    inception.to('cuda')
    inception.eval()
    batch_size=128
    real_images = torch.from_numpy(real_images) # conversion from numpy for passing it to the model
    gen_images  = torch.from_numpy(gen_images)
    def activations_model(images):
        embedds = []
        l= images.shape[0]
        with torch.no_grad(): # to prevent gradient calculation
            for i in range(0, l , batch_size):
                batch_images = images[i:i+batch_size].to('cuda')
                batch_images = batch_images.permute(0,3,1,2) # convert to (B,C,H,W)
                batch_activations = inception(batch_images)
                embedds.append(batch_activations.cpu().numpy())
        embedds = np.concatenate(embedds) # convert the list of numpy array to numpy array
        return embedds

    real_img_embed = activations_model(real_images) # embedding for real images

    gen_img_embed = activations_model(gen_images) #embedding for generated images
    
    # Now let us calculate the statistics of the embeddings
    ################################## real image #####################################
    mu_real = np.mean(real_img_embed, axis=0) 
    C_real = np.cov(real_img_embed, rowvar=False)
    ##############################################################################

    ################################# generated image ################################
    mu_gen = np.mean(gen_img_embed, axis=0), 
    C_gen  = np.cov(gen_img_embed, rowvar=False)
    ##################################################################################

    ##### from definition of FID score see : https://en.wikipedia.org/wiki/Fr%C3%A9chet_inception_distance #########
    ########### d^2 = ||mu_1 – mu_2||^2 + Tr(C_1 + C_2 – 2*sqrt(C_1*C_2)) ####################################
    diff = np.dot(mu_real - mu_gen,(mu_real - mu_gen).T)
    covmean = linalg.sqrtm(C_real.dot(C_gen))
    if np.iscomplexobj(covmean): ### to avoid the complex number in the calculation of the FID score
        covmean = covmean.real
    fid_score = diff + np.trace(C_real + C_gen - 2.0 * covmean)
    ################################################################################################################

    return fid_score

####################### Loading the real and generated images from respective directories ###########################
real_images = []
c=0
for file_path in glob.glob(os.path.join(real_image_dir, '*.png')):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)
    real_images.append(img)
    c+=1
    if c>7000:
        break
real_images = np.array(real_images)

gen_images = []
for file_path in glob.glob(os.path.join(generated_image_dir, '*.png')):
    img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    img = preprocess_image(img)
    gen_images.append(img)
gen_images = np.array(gen_images)
###################################################################################################################

fid_score = fid_score(real_images, gen_images)
print('FID score:', fid_score)
