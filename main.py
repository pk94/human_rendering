import torchvision.models as models
import pytorch_lightning as pl
from feature_net import *
from losses import *
from datasets import *
from textures import MapDensePoseTexModule
from torch.utils.data import DataLoader
import cv2
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torchvision.transforms as transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class HumanRendering(pl.LightningModule):
    def __init__(self, checkpoint_path, batch_size=64):
        super(HumanRendering, self).__init__()
        self.checkpoint_path = checkpoint_path
        self.save_hyperparameters()
        self.batch_size = batch_size
        self.texture_mapper = MapDensePoseTexModule(256).eval()
        self.feature_net = FeatureNet(3, 16).to(device)
        self.render_net = RenderNet(16, 3).to(device)
        self.discriminators_feature = [PatchDiscriminator(3).to(device), PatchDiscriminator(3).to(device),
                                       PatchDiscriminator(3).to(device)]
        self.discriminators_render = [PatchDiscriminator(3).to(device), PatchDiscriminator(3).to(device),
                                      PatchDiscriminator(3).to(device)]
        self.vgg19 = models.vgg19(pretrained=True).to(device).eval()
        self.face_detector = MTCNN(image_size=160, margin=15, device='cuda').eval()
        self.face_rec = InceptionResnetV1(pretrained='vggface2').eval().cuda()
        self.hook = SaveOutput()
        self.vgg19_normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        self.model_num = 0
        # self.contextual_loss = cl.ContextualLoss()
        hook_handles = []
        for idx, layer in enumerate(self.vgg19.features):
            if isinstance(layer, nn.ReLU):
                handle = layer.register_forward_hook(self.hook)
                hook_handles.append(handle)
        # self.total_loss = {
        #     'generator': 0,
        #     'discriminator': 0
        # }
        # self.losses = {
        #     'generator': [],
        #     'discriminator': []
        # }

    def forward(self, train_batch):
        feature_out = self.feature_net(train_batch['sample']['texture'])
        feature_out_tex = self.apply_texture(feature_out, train_batch['target']['instances'],
                                             train_batch['target']['uv'])
        render_out = self.render_net(feature_out_tex)
        return feature_out, feature_out_tex, render_out

    def training_step(self, train_batch, batch_nb, optimizer_idx):
        feature_out, feature_out_tex, render_out = self.forward(train_batch)
        target_applied_texture = self.apply_texture(train_batch['target']['texture'],
                                                    train_batch['target']['instances'], train_batch['target']['uv'])
        #
        # generated_image = target_applied_texture[0].detach().permute((1, 2, 0)).add(1).true_divide(2).mul(
        #     255).cpu().numpy().astype(np.uint8)
        # cv2.imwrite(f'images/target_applied_texture.jpg', generated_image)

        # train generator
        if optimizer_idx == 0:
            loss_inpainting = inpainting_loss(feature_out, train_batch['sample']['texture'],
                                              train_batch['target']['texture'])
            loss_adversarial_feature = adversarial_loss(self.discriminators_feature, target_applied_texture,
                                                        feature_out_tex, is_discriminator=False, is_feature=True)
            loss_adversarial_render = adversarial_loss(self.discriminators_render, train_batch['target']['image'],
                                                       render_out, is_discriminator=False, is_feature=False)
            target_vgg = self.vgg19_normalize(torch.clone(train_batch['target']['image']).add(1).true_divide(2))
            generated_vgg = self.vgg19_normalize(torch.clone(render_out).add(1).true_divide(2))
            self.vgg19(target_vgg)
            ground_truth_activations = self.hook.outputs
            self.hook.clear()
            self.vgg19(generated_vgg)
            generated_activations = self.hook.outputs
            self.hook.clear()
            loss_perceptual = perceptual_loss(ground_truth_activations, generated_activations)
            loss_identity = face_identity_loss(self.face_rec, train_batch['sample']['image'],
                                               train_batch['sample']['instances'], train_batch['target']['image'],
                                               train_batch['target']['instances'], render_out, self.face_detector)
            # contextual_loss = self.contextual_loss(train_batch['sample']['image'], train_batch['target']['image'])
            # fft_loss = fourier_loss(render_out, train_batch['target']['image'])
            total_loss = (2 * loss_adversarial_render + loss_adversarial_feature + 2 * loss_inpainting
                          + 5 * loss_perceptual + 2 * loss_identity) / 5
            # self.total_loss['generator'] = total_loss
            self.log('gen_adversarial_render', loss_adversarial_render)
            self.log('gen_adversarial_feature', loss_adversarial_feature)
            return total_loss

        # train discriminator
        if optimizer_idx == 1:
            loss_adversarial_feature = adversarial_loss(self.discriminators_feature, target_applied_texture,
                                                        feature_out_tex, is_discriminator=True, is_feature=True)
            # self.total_loss['discriminator'] += loss_adversarial_feature
            self.log('disc_adversarial_feature', loss_adversarial_feature)
            return loss_adversarial_feature

        if optimizer_idx == 2:
            loss_adversarial_render = adversarial_loss(self.discriminators_render, train_batch['target']['image'],
                                                       render_out, is_discriminator=True, is_feature=False)
            # self.total_loss['discriminator'] += loss_adversarial_render
            self.log('disc_adversarial_render', loss_adversarial_render)
            return loss_adversarial_render

    def validation_step(self, val_batch, batch_nb):
        feature_out, feature_out_tex, render_out = self.forward(val_batch)
        for idx in range(feature_out.shape[0]):
            path = f'tests/test_{self.model_num}/pair{batch_nb * 4 + idx}'
            if not os.path.exists(path):
                os.makedirs(path)
            self.save_img(render_out[idx], path + '/rendered.jpg')
            self.save_img(val_batch['sample']['image'][idx], path + '/source.jpg')
            self.save_img(val_batch['sample']['texture'][idx], path + '/source_texture.jpg')
            self.save_uv(val_batch['sample']['uv'][idx], path + '/source_pose.jpg')
            self.save_img(val_batch['target']['image'][idx], path + '/target.jpg')
            self.save_uv(val_batch['target']['uv'][idx], path + '/target_pose.jpg')
            self.save_img(feature_out[idx], path + '/generated_texture.jpg', False)
            self.save_img(feature_out_tex[idx], path + '/texture_applied.jpg', False)

    def on_validation_end(self):
        checkpoints_num = [int(f.split("_")[-1][:-5]) for f in os.listdir(self.checkpoint_path) if os.path.isfile(os.path.join(self.checkpoint_path, f))]
        self.model_num = max(checkpoints_num) + 1
        self.trainer.save_checkpoint(self.checkpoint_path + f"checkpoint_{self.model_num}.ckpt")

    def configure_optimizers(self):
        lr = 0.0001
        b1 = 0.4
        b2 = 0.9

        opt_gen = torch.optim.Adam(list(self.feature_net.parameters()) + list(self.render_net.parameters()),
                                   lr=lr, betas=(b1, b2))
        opt_disc_feature = torch.optim.Adam(list(self.discriminators_feature[0].parameters()) +
                                            list(self.discriminators_feature[1].parameters()) +
                                            list(self.discriminators_feature[2].parameters()), lr=lr, betas=(b1, b2))
        opt_disc_render = torch.optim.Adam(list(self.discriminators_render[0].parameters()) +
                                           list(self.discriminators_render[1].parameters()) +
                                           list(self.discriminators_render[2].parameters()), lr=lr, betas=(b1, b2))
        lr_lambda = lambda epoch: 0.999
        scheduler_gen = torch.optim.lr_scheduler.MultiplicativeLR(opt_gen, lr_lambda)
        scheduler_disc_feature = torch.optim.lr_scheduler.MultiplicativeLR(opt_disc_feature, lr_lambda)
        scheduler_disc_render = torch.optim.lr_scheduler.MultiplicativeLR(opt_disc_render, lr_lambda)
        return [opt_gen, opt_disc_feature, opt_disc_render], [scheduler_gen, scheduler_disc_feature,
                                                              scheduler_disc_render]

    def train_dataloader(self):
        dataset = DeepFashionCSV("csv_files/train.csv")
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=8)

    def val_dataloader(self):
        dataset = DeepFashionCSV("csv_files/test.csv")
        return DataLoader(dataset, batch_size=self.batch_size, num_workers=8)

    def save_img(self, tensor, savepath, more_channels=True):
        img = tensor.detach().permute((1, 2, 0)).add(1).true_divide(2).mul(255).cpu().numpy().astype(np.uint8) if \
            more_channels else tensor.detach().permute((1, 2, 0)).add(1).true_divide(2).mul(255).cpu().numpy().astype(np.uint8)[:, :, :3]
        cv2.imwrite(savepath, img)

    def save_uv(self, tensor, savepath):
        u, v = tensor.detach().cpu().numpy()
        img = np.stack((u, v, (u + v) * 0.5), axis=-1).astype(np.uint8)
        cv2.imwrite(savepath, img)

    def apply_texture(self, generated_textures, target_instances, target_uv_maps):
        uv_tensor = target_uv_maps.byte()
        instances_tensor = target_instances.byte()
        iuv_tensor = torch.cat((instances_tensor[:, None, :, :], uv_tensor), dim=1)
        output = self.texture_mapper(generated_textures, iuv_tensor)
        return output


model = HumanRendering(checkpoint_path="checkpoints/", batch_size=8)

trainer = pl.Trainer(gpus=1, auto_select_gpus=True, val_check_interval=0.5, max_epochs=1000, resume_from_checkpoint="/home/pkowaleczko/projects/human_rendering/checkpoints/checkpoint_115.ckpt")
trainer.fit(model)

# trainer.fit(model)