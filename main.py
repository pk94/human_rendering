import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import torchvision.models as models
import pytorch_lightning as pl
from feature_net import *
from losses import *
from datasets import *
from textures import MapDensePoseTexModule
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HumanRendering(pl.LightningModule):

    def __init__(self, data_path, batch_size=64):
        super(HumanRendering, self).__init__()
        self.save_hyperparameters()
        self.data_path = data_path
        self.batch_size = batch_size
        self.texture_mapper = MapDensePoseTexModule(256).eval()
        self.feature_net = FeatureNet(3, 16).to(device)
        self.render_net = RenderNet(16, 3).to(device)
        self.discriminators = [PatchDiscriminator(6).to(device), PatchDiscriminator(6).to(device),
                               PatchDiscriminator(6).to(device)]
        self.vgg19 = models.vgg19(pretrained=True).to(device).eval()
        self.face_detector = MTCNN(image_size=160, margin=15, device='cuda').eval()
        self.face_rec = InceptionResnetV1(pretrained='vggface2').eval().cuda()
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
        feature_out_tex = self.apply_texture(feature_out, train_batch['target']['instances'],
                                             train_batch['target']['uv'])
        render_out = self.render_net(feature_out_tex)
        return feature_out, feature_out_tex, render_out

    def training_step(self, train_batch, batch_nb, optimizer_idx):
        feature_out, feature_out_tex, render_out = self.forward(train_batch)
        # train generator
        target_applied_texture = self.apply_texture(train_batch['target']['texture'],
                                                    train_batch['target']['instances'], train_batch['target']['uv'])
        real_gan_input = torch.cat((train_batch['target']['image'], target_applied_texture), 1)
        fake_gan_input = torch.cat((render_out, feature_out_tex[:, :3, :, :]), 1)
        generated_image = real_gan_input[0][:3, :, :].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(255).cpu().numpy(). \
            astype(np.uint8)
        cv2.imwrite(f'images/a.jpg', generated_image[:, :, :3])
        generated_image = real_gan_input[0][3:, :, :].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(
            255).cpu().numpy(). \
            astype(np.uint8)
        cv2.imwrite(f'images/b.jpg', generated_image[:, :, :3])
        generated_image = fake_gan_input[0][:3, :, :].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(
            255).cpu().numpy(). \
            astype(np.uint8)
        cv2.imwrite(f'images/c.jpg', generated_image[:, :, :3])
        generated_image = fake_gan_input[0][3:, :, :].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(
            255).cpu().numpy(). \
            astype(np.uint8)
        cv2.imwrite(f'images/d.jpg', generated_image[:, :, :3])

        if optimizer_idx == 0:
            loss_inpainting = inpainting_loss(feature_out, train_batch['sample']['texture'],
                                              train_batch['target']['texture'])
            loss_adversarial = adversarial_loss(self.discriminators, real_gan_input, fake_gan_input,
                                                is_discriminator=False)
            self.vgg19(train_batch['target']['image'])
            ground_truth_activations = self.hook.outputs
            self.hook.clear()
            self.vgg19(render_out)
            generated_activations = self.hook.outputs
            self.hook.clear()
            loss_perceptual = perceptual_loss(ground_truth_activations, generated_activations)
            loss_identity = face_identity_loss(self.face_rec, train_batch['sample']['image'],
                                               train_batch['sample']['instances'], train_batch['target']['image'],
                                               train_batch['target']['instances'], render_out, self.face_detector)
            total_loss = (loss_adversarial + loss_inpainting + 5 * loss_perceptual + 2 * loss_identity) / 4
            self.total_loss['generator'] = total_loss
            return total_loss

        if optimizer_idx == 1:
            loss_adversarial = adversarial_loss(self.discriminators, real_gan_input, fake_gan_input,
                                                is_discriminator=True)
            total_loss = loss_adversarial
            self.total_loss['discriminator'] = total_loss
            self.from_batch_generate_image(train_batch)
            return total_loss

    def configure_optimizers(self):
        lr = 0.002
        b1 = 0.5
        b2 = 0.9

        opt_gen = torch.optim.Adam(list(self.feature_net.parameters()) + list(self.render_net.parameters()),
                                   lr=lr, betas=(b1, b2))
        opt_disc = torch.optim.Adam(list(self.discriminators[0].parameters()) +
                                    list(self.discriminators[1].parameters()) +
                                    list(self.discriminators[2].parameters()), lr=lr, betas=(b1, b2))
        lr_lambda = lambda epoch: 0.95
        scheduler_gen = torch.optim.lr_scheduler.MultiplicativeLR(opt_gen, lr_lambda)
        scheduler_disc = torch.optim.lr_scheduler.MultiplicativeLR(opt_disc, lr_lambda)
        return [opt_gen, opt_disc], [scheduler_gen, scheduler_disc]

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
        plt.savefig('images/losses.jpg')

    def from_batch_generate_image(self, batch):
        self.set_models_mode('eval', False)
        textures, textures_applied, rendered = self.forward(batch)

        generated_image = textures[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(255).cpu().numpy(). \
            astype(np.uint8)
        cv2.imwrite(f'images/texture.jpg', generated_image[:, :, :3])

        generated_image = textures_applied[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(255).cpu().numpy().\
            astype(np.uint8)
        cv2.imwrite(f'images/genrated_texture.jpg', generated_image[:, :, :3])

        original_im = batch['sample']['image']
        generated_image = original_im[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(
            255).cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'images/original.jpg', generated_image)

        original_im = batch['sample']['texture']
        generated_image = original_im[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(
            255).cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'images/original_texture.jpg', generated_image)

        original_im = batch['target']['image']
        generated_image = original_im[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(
            255).cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'images/target.jpg', generated_image)

        original_im = batch['target']['texture']
        generated_image = original_im[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(
            255).cpu().numpy().astype(np.uint8)
        cv2.imwrite(f'images/target_texture.jpg', generated_image)

        generated_image = rendered[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(255).cpu().numpy().\
            astype(np.uint8)
        cv2.imwrite(f'images/rendered.jpg', generated_image)
        self.set_models_mode('train', False)

    def apply_texture(self, generated_textures, target_instances, target_uv_maps):
        uv_tensor = target_uv_maps.byte()
        instances_tensor = target_instances.byte()
        iuv_tensor = torch.cat((instances_tensor[:, None, :, :], uv_tensor), dim=1)
        output = self.texture_mapper(generated_textures, iuv_tensor)
        return output

    def set_models_mode(self, mode, only_discriminators):
        if mode == 'eval':
            if not only_discriminators:
                self.feature_net.eval()
                self.render_net.eval()
            for disc in self.discriminators:
                disc.eval()
        if mode == 'train':
            if not only_discriminators:
                self.feature_net.train()
                self.render_net.train()
            for disc in self.discriminators:
                disc.train()


model = HumanRendering('/home/pkowaleczko/datasets/deepfashion/DeepfashionProcessed', batch_size=8)

trainer = pl.Trainer(gpus=1, auto_select_gpus=True, max_epochs=10000)
trainer.fit(model)
