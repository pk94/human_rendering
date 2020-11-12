import torchvision.models as models
import pytorch_lightning as pl
from feature_net import *
from losses import *
from datasets import *
from textures import MapDensePoseTexModule
import matplotlib.pyplot as plt
from torchvision.utils import save_image
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HumanRendering(pl.LightningModule):

    def __init__(self, data_path, batch_size=64):
        super(HumanRendering, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.texture_mapper = MapDensePoseTexModule(256)
        self.feature_net = FeatureNet(3, 3).to(device)
        self.render_net = RenderNet(3, 3).to(device)
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

    def forward(self, train_batch):
        feature_out = self.feature_net(train_batch['sample']['texture'])
        # feature_out_tex = self.apply_texture(feature_out, train_batch['target']['instances'],
        #                                      train_batch['target']['uv'])
        feature_out_tex = self.apply_texture(train_batch['sample']['texture'], train_batch['sample']['instances'],
                                             train_batch['sample']['uv'])
        render_out = self.render_net(feature_out_tex)
        return feature_out, feature_out_tex, render_out

    def training_step(self, train_batch, batch_nb, optimizer_idx):
        feature_out, feature_out_tex, render_out = self.forward(train_batch)
        # train generator
        if optimizer_idx == 0:
            loss_inpainting = inpainting_loss(feature_out, train_batch['sample']['texture'],
                                              train_batch['sample']['texture'])
            loss_adversarial_feature = adversarial_loss(self.discriminators_feature, train_batch['sample']['image'],
                                                        feature_out_tex, is_discriminator=False, is_feature=True)
            loss_adversarial_render = adversarial_loss(self.discriminators_render, train_batch['sample']['image'],
                                                       render_out, is_discriminator=False, is_feature=False)
            self.vgg19.forward(train_batch['sample']['image'])
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
            loss_adversarial_feature = adversarial_loss(self.discriminators_feature, train_batch['sample']['image'],
                                                        feature_out_tex, is_discriminator=True, is_feature=True)
            self.total_loss['discriminator'] += loss_adversarial_feature
            return loss_adversarial_feature

        if optimizer_idx == 2:
            loss_adversarial_render = adversarial_loss(self.discriminators_render, train_batch['sample']['image'],
                                                       render_out, is_discriminator=True, is_feature=False)
            self.total_loss['discriminator'] += loss_adversarial_render
            self.on_training_step_end()
            self.from_batch_generate_image(train_batch)
            return loss_adversarial_render

    def configure_optimizers(self):
        lr = 0.00005
        b1 = 0.5
        b2 = 0.5

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
        dataset = DeepFashionDataset(self.data_path)
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
        output = self.forward(batch)[2]
        generated_image = output[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(255).cpu().numpy().astype(np.uint8)
        save_image(output[0], 'generated_torch.jpg')
        cv2.imwrite('genrated.jpg', generated_image)

    def apply_texture(self, generated_textures, target_instances, target_uv_maps):
        uv_tensor = target_uv_maps.byte()
        instances_tensor = target_instances.byte()
        iuv_tensor = torch.cat((instances_tensor[:, None, :, :], uv_tensor), dim=1)
        output = self.texture_mapper(generated_textures, iuv_tensor)
        return output

model = HumanRendering('/home/pawel/Datasets/deepfashion-processed/deepfashion_sample', batch_size=2)
trainer = pl.Trainer(gpus=1, auto_select_gpus=True)
trainer.fit(model)
