import torch
import torch.nn as nn
import torch.nn.functional as F
from render_net import *


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []

def l1_distance(tensor, tensor_sub):
    diff_tensor = torch.abs(tensor - tensor_sub)
    return torch.sum(diff_tensor)



def perceptual_loss(model, ground_truth, generated):
    save_output = SaveOutput()
    hook_handles = []
    for layer in model.modules():
        if isinstance(layer, nn.ReLU):
            handle = layer.register_forward_hook(save_output)
            hook_handles.append(handle)
    model(ground_truth)
    ground_truth_activations = save_output.outputs
    save_output.clear()
    model(generated)
    generated_activations = save_output.outputs
    loss = 0
    for ground_truth_activation, generated_activation in zip(ground_truth_activations, generated_activations):
        num_elements = 1
        for dim in ground_truth_activation.size()[1:]:
            num_elements *= dim
        loss += l1_distance(ground_truth_activation, generated_activation) / num_elements
    return loss


def adversarial_loss(models, real_image, fake_image, is_discriminator):
    loss_function = nn.BCELoss()
    loss = 0
    activation = nn.Sigmoid()
    for idx, model in enumerate(models):
        if is_discriminator:
            real_image_down = F.interpolate(real_image, scale_factor=1 / (2 ** idx), mode='bilinear')
            disc_real_out = model(real_image_down)
        fake_image_down = F.interpolate(fake_image, scale_factor=1 / (2 ** idx), mode='bilinear')
        disc_fake_out = model(fake_image_down)
        if is_discriminator:
            real_loss = loss_function(activation(disc_real_out), torch.ones_like(disc_real_out))
            fake_loss = loss_function(activation(disc_fake_out), torch.zeros_like(disc_fake_out))
            loss += (real_loss + fake_loss) / 2
        else:
            fake_loss = loss_function(activation(disc_fake_out), torch.ones_like(disc_fake_out))
            loss += fake_loss
    return loss / len(models)

def inpainting_loss(features, input_texture, target_texture):
    dist_in_gen = l1_distance(input_texture[:, :3, :, :], features[:, :3, :, :])
    dist_tar_gen = l1_distance(target_texture[:, :3, :, :], features[:, :3, :, :])
    return dist_in_gen + dist_tar_gen