# Segmentation_models_for_Diffusion
The model can be trained using the command
```console
python train.py
```
All the models except the LinkNet model is trained using 32\*32 verison of fashion MNIST and MNIST dataset. For sampling the image, give the path to the model weight and the path to folder where image is to be generated. Make sure that correct Model definition is given in file.
```console
python sample.py
```
To calculate the FID score, give the correct path to the generated and real images in the file and run the command
```console
python FID.py
```