import torchvision.models as models
import pytorch_lightning as pl
from feature_net import *
from losses import *
from datasets import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HumanRendering(pl.LightningModule):

    def __init__(self, data_path, batch_size=64):
        super(HumanRendering, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_net = FeatureNet(3, 16)
        self.render_net = RenderNet(16, 3)
        self.discriminators = [PatchDiscriminator(3), PatchDiscriminator(3), PatchDiscriminator(3)]
        self.vgg19 = models.vgg19(pretrained=True)

    def forward(self, x):
        feature_out = self.feature_net(x)
        render_out = self.render_net(feature_out)
        return feature_out, render_out

    def training_step(self, train_batch, batch_nb, optimizer_idx):
        input_batch = train_batch[:int(train_batch.shape[0] / 2), :, :, :]
        target_batch = train_batch[int(train_batch.shape[0] / 2):, :, :, :]

        # train generator
        if optimizer_idx == 0 or optimizer_idx == 1:
            feature_out, render_out = self.forward(input_batch)
            loss_inpainting = inpainting_loss(feature_out, input_batch, target_batch)
            loss_adversarial = adversarial_loss(self.discriminators, input_batch, target_batch, is_discriminator=False)
            loss_perceptual = perceptual_loss(self.vgg19, target_batch, render_out)
            print(f'\nGen loss: {loss_inpainting}  {loss_adversarial}  {loss_perceptual}')
            return loss_inpainting + loss_adversarial + loss_perceptual

        # train discriminator
        if optimizer_idx == 2 or optimizer_idx == 3 or optimizer_idx == 4:
            loss_adversarial = adversarial_loss(self.discriminators, input_batch, target_batch, is_discriminator=True)
            print(f'Disc loss: {loss_adversarial}')
            return loss_adversarial

    def configure_optimizers(self):
        lr = 0.0002
        b1 = 0.5
        b2 = 0

        opt_feature_net = torch.optim.Adam(self.feature_net.parameters(), lr=lr, betas=(b1, b2))
        opt_render_net = torch.optim.Adam(self.render_net.parameters(), lr=lr, betas=(b1, b2))
        opt_dicriminator_0 = torch.optim.Adam(self.discriminators[0].parameters(), lr=lr, betas=(b1, b2))
        opt_dicriminator_1 = torch.optim.Adam(self.discriminators[1].parameters(), lr=lr, betas=(b1, b2))
        opt_dicriminator_2 = torch.optim.Adam(self.discriminators[2].parameters(), lr=lr, betas=(b1, b2))
        return [opt_feature_net, opt_render_net, opt_dicriminator_0, opt_dicriminator_1, opt_dicriminator_2], []

    def train_dataloader(self):
        dataset = VideoDataset(self.data_path)
        return DataLoader(dataset, batch_size=self.batch_size)


model = HumanRendering('data/train/')
trainer = pl.Trainer()
trainer.fit(model)

