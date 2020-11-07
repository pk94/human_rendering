import torchvision.models as models
import pytorch_lightning as pl
from feature_net import *
from losses import *
from datasets import *
import matplotlib.pyplot as plt
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HumanRendering(pl.LightningModule):

    def __init__(self, data_path, batch_size=64):
        super(HumanRendering, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.feature_net = FeatureNet(3, 16).to(device)
        self.render_net = RenderNet(16, 3).to(device)
        self.discriminators_feature = [PatchDiscriminator(3).to(device), PatchDiscriminator(3).to(device),
                                       PatchDiscriminator(3).to(device)]
        self.discriminators_render = [PatchDiscriminator(3).to(device), PatchDiscriminator(3).to(device),
                                      PatchDiscriminator(3).to(device)]
        self.vgg19 = models.vgg19(pretrained=True).to(device)
        self.hook = SaveOutput()
        hook_handles = []
        for layer in self.vgg19.modules():
            if isinstance(layer, nn.ReLU):
                handle = layer.register_forward_hook(self.hook)
                hook_handles.append(handle)
        self.total_loss = {
            'generator': 0,
            'discriminator': 0
        }
        self.losses = {
            'generator': [],
            'discriminator': []
        }

    def forward(self, x):
        feature_out = self.feature_net(x)
        render_out = self.render_net(feature_out)
        return feature_out, render_out

    def training_step(self, train_batch, batch_nb, optimizer_idx):
        input_batch = train_batch[:int(train_batch.shape[0] / 2), :, :, :]
        target_batch = train_batch[int(train_batch.shape[0] / 2):, :, :, :]
        feature_out, render_out = self.forward(input_batch)
        # train generator
        if optimizer_idx == 0:
            loss_inpainting = inpainting_loss(feature_out, input_batch, target_batch)
            loss_adversarial_feature = adversarial_loss(self.discriminators_feature, target_batch, feature_out,
                                                        is_discriminator=False, is_feature=True)
            loss_adversarial_render = adversarial_loss(self.discriminators_render, target_batch, render_out,
                                                       is_discriminator=False, is_feature=False)
            self.vgg19.forward(target_batch)
            ground_truth_activations = self.hook.outputs
            self.hook.clear()
            self.vgg19.forward(render_out)
            generated_activations = self.hook.outputs
            self.hook.clear()
            loss_perceptual = perceptual_loss(ground_truth_activations, generated_activations)
            total_loss = (loss_adversarial_render + loss_adversarial_feature + loss_inpainting + loss_perceptual) / 4
            self.total_loss['generator'] = total_loss
            return total_loss

        # train discriminator
        if optimizer_idx == 1:
            loss_adversarial_feature = adversarial_loss(self.discriminators_feature, target_batch, feature_out,
                                                        is_discriminator=True, is_feature=True)
            self.total_loss['discriminator'] += loss_adversarial_feature
            return loss_adversarial_feature

        if optimizer_idx == 2:
            loss_adversarial_render = adversarial_loss(self.discriminators_render, target_batch, render_out,
                                                       is_discriminator=True, is_feature=False)
            self.total_loss['discriminator'] += loss_adversarial_render
            self.on_training_step_end()
            self.from_batch_generate_image(target_batch)
            return loss_adversarial_render

    def configure_optimizers(self):
        lr = 0.0001
        b1 = 0.5
        b2 = 0.99

        opt_gen = torch.optim.Adam(list(self.feature_net.parameters()) + list(self.render_net.parameters()),
                                   lr=lr, betas=(b1, b2))
        opt_disc_feature = torch.optim.Adam(list(self.discriminators_feature[0].parameters()) +
                                            list(self.discriminators_feature[1].parameters()) +
                                            list(self.discriminators_feature[2].parameters()), lr=lr, betas=(b1, b2))
        opt_disc_render = torch.optim.Adam(list(self.discriminators_render[0].parameters()) +
                                           list(self.discriminators_render[1].parameters()) +
                                           list(self.discriminators_render[2].parameters()), lr=lr, betas=(b1, b2))
        lr_lambda = lambda epoch: 0.99
        scheduler_gen = torch.optim.lr_scheduler.MultiplicativeLR(opt_gen, lr_lambda)
        scheduler_disc_feature = torch.optim.lr_scheduler.MultiplicativeLR(opt_disc_feature, lr_lambda)
        scheduler_disc_render = torch.optim.lr_scheduler.MultiplicativeLR(opt_disc_render, lr_lambda)
        return [opt_gen, opt_disc_feature, opt_disc_render], \
               [scheduler_gen, scheduler_disc_feature, scheduler_disc_render]

    def train_dataloader(self):
        dataset = VideoDataset(self.data_path)
        return DataLoader(dataset, batch_size=self.batch_size)

    def on_training_step_end(self):
        self.losses['generator'].append(self.total_loss['generator'])
        self.losses['discriminator'].append(self.total_loss['discriminator'])
        self.total_loss['discriminator'] = 0
        plt.clf()
        plt.title('Losses')
        plt.plot(self.losses['generator'], label='Generator loss')
        plt.plot(self.losses['discriminator'], label='Discriminator loss')
        plt.legend()
        plt.savefig('losses.jpg')

    def from_batch_generate_image(self, batch):
        input_img = batch[0, :, :, :].unsqueeze(0)
        generated_img = self.forward(input_img)[1]
        save_image(generated_img, 'generated_image.jpg')


model = HumanRendering('data/train/', batch_size=16)
trainer = pl.Trainer(gpus=1, auto_select_gpus=True)
trainer.fit(model)
