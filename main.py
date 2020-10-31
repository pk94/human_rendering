import pickle
import numpy as np
from torchvision import transforms
import torchvision.models as models
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from feature_net import *
from losses import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HumanRendering(pl.LightningModule):

    def __init__(self):
        super(HumanRendering, self).__init__()

        self.feature_net = FeatureNet(3, 16)
        self.render_net = RenderNet(16, 3)
        self.discriminators = [PatchDiscriminator(3), PatchDiscriminator(3), PatchDiscriminator(3)]
        self.vgg19 = models.vgg19(pretrained=True)

    def forward(self, x):
        feature_out = self.feature_net(x)
        render_out = self.render_net(feature_out)
        vgg_out = self.vgg19(render_out)
        return feature_out, render_out, vgg_out

    def training_step(self, train_batch, batch_nb, optimizer_i):
        input_batch, target_batch = random_split(train_batch, [train_batch.size()[0]/2, train_batch.size()[0]/2])

        # train generator
        if optimizer_i == 0:
            feature_out, render_out, vgg_out = self.forward(input_batch)
            loss_inpainting = inpainting_loss(feature_out, input_batch, target_batch)
            loss_adversarial = adversarial_loss(self.discriminators, input_batch, target_batch, is_discriminator=False)
            loss_perceptual = perceptual_loss(self.vgg19, target_batch, vgg_out)
            return loss_inpainting + loss_adversarial + loss_perceptual

        # train discriminator
        if optimizer_i == 1:
            loss_adversarial = adversarial_loss(self.discriminators, input_batch, target_batch, is_discriminator=True)
            return loss_adversarial

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0

        opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
        return [opt_g, opt_d], []

    def train_dataloader(self):
        pass

with open('data/train.pkl', 'rb') as f:
    dataset = pickle.load(f)

images = dataset['rgb']  # list of images, not equal in size
textures = dataset['tex']  # list of textures, so they can be processed as np.ndarray
# img = torch.from_numpy(np.rollaxis(textures[0][None, :, :], 3, 1)).float().to(device)
# model_feature = FeatureNet(3, 16).cuda()
# a = model_feature.forward(img)
# model_render = RenderNet(16, 3).cuda()
# b = model_render.forward(a)
# vgg19 = models.vgg19(pretrained=True).cuda()
# perceptual_loss(vgg19, img, b)
# models_array = [PatchDiscriminator(3).cuda(), PatchDiscriminator(3).cuda(), PatchDiscriminator(3).cuda()]
# adversarial_loss(models_array, img, b)
# summary(vgg19, (3, 128, 128))
