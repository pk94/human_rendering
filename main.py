import pickle
import numpy as np
from feature_net import *
from render_net import *
from torchsummary import summary
import torchvision.models as models
import cv2
from losses import *
import os

with open('data/train.pkl', 'rb') as f:
	dataset = pickle.load(f)

images = dataset['rgb'] # list of images, not equal in size
textures = dataset['tex'] # list of textures, so they can be processed as np.ndarray
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
img = torch.from_numpy(np.rollaxis(textures[0][None, :, :], 3, 1)).float().to(device)
model_feature = FeatureNet(3, 16).cuda()
a = model_feature.forward(img)
model_render = RenderNet(16, 3).cuda()
b = model_render.forward(a)
vgg19 = models.vgg19(pretrained=True).cuda()
perceptual_loss(vgg19, img, b)
models_array = [PatchDiscriminator(3).cuda(), PatchDiscriminator(3).cuda(), PatchDiscriminator(3).cuda()]
adversarial_loss(models_array, img, b)
summary(vgg19, (3, 128, 128))
