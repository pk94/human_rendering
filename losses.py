from render_net import *
import torch
from facenet_pytorch import extract_face
import cv2
from facenet_pytorch import InceptionResnetV1


class SaveOutput:
    def __init__(self):
        self.outputs = []

    def __call__(self, module, module_in, module_out):
        self.outputs.append(module_out)

    def clear(self):
        self.outputs = []


def l1_distance(tensor, tensor_sub):
    diff_tensor = torch.abs(tensor - tensor_sub)
    return torch.mean(diff_tensor)


def perceptual_loss(ground_truth_activations, generated_activations):
    loss = 0
    for ground_truth_activation, generated_activation in zip(ground_truth_activations, generated_activations):
        num_elements = 1
        for dim in ground_truth_activation.size()[1:]:
            num_elements *= dim
        loss += l1_distance(ground_truth_activation, generated_activation) / len(ground_truth_activations)
    return loss


def adversarial_loss(models, real_image, fake_image, is_discriminator):
    loss_function = nn.BCELoss()
    loss = 0
    activation = nn.Sigmoid()
    for idx, model in enumerate(models):
        if is_discriminator:
            real_image_down = F.interpolate(real_image, scale_factor=1 / (2 ** idx), mode='bilinear')
            disc_real_out = activation(model(real_image_down))
        fake_image_down = F.interpolate(fake_image, scale_factor=1 / (2 ** idx), mode='bilinear')
        disc_fake_out = activation(model(fake_image_down))
        if is_discriminator:
            real_loss = loss_function(disc_real_out, torch.ones_like(disc_real_out))
            fake_loss = loss_function(disc_fake_out, torch.zeros_like(disc_fake_out))
            loss += (real_loss + fake_loss) / 2
        else:
            fake_loss = loss_function(disc_fake_out, torch.ones_like(disc_fake_out))
            loss += fake_loss
    return loss / len(models)


def inpainting_loss(features, input_texture, target_texture):
    dist_in_gen = l1_distance((1 - calculate_texture_mask(input_texture)) * input_texture, features[:, :3, :, :])
    dist_tar_gen = l1_distance((1 - calculate_texture_mask(target_texture)) * target_texture, features[:, :3, :, :])
    return dist_in_gen + dist_tar_gen

def calculate_texture_mask(texture):
    mask = torch.ones_like(texture)
    indicies = texture != texture[0, 0, 0, 0]
    mask[indicies] = 0
    return mask


def face_identity_loss(model, source_images, source_instances, target_images, target_instances, rendered_images,
                       face_detector):
    face_match_counter = 0
    loss = 0
    model = model.eval()
    for source_image, source_instance, target_image, target_instance, rendered_image in \
            zip(source_images, source_instances, target_images, target_instances, rendered_images):
        source_face = crop_and_resize_face(source_image, source_instance, face_detector, False, rendered_image)
        target_face = crop_and_resize_face(target_image, target_instance, face_detector, True, rendered_image)
        if not isinstance(source_face, torch.Tensor) or not isinstance(target_face, torch.Tensor):
            pass
        else:
            face_match_counter += 1
            source_face = torch.unsqueeze(source_face, 0)
            target_face = torch.unsqueeze(target_face, 0)
            source_face_flipped = torch.flip(source_face, (3,))
            target_face_flipped = torch.flip(target_face, (3,))
            source_face_embedding = torch.cat((model(source_face), model(source_face_flipped)), -1)
            target_face_embedding = torch.cat((model(target_face), model(target_face_flipped)), -1)
            cos = nn.CosineSimilarity(dim=1, eps=1e-6)
            loss += 1 - cos(source_face_embedding, target_face_embedding)
    return loss

def crop_and_resize_face(source_image, instances, face_detector, use_rendered, rendered_image, target_shape=(160, 160)):
    face_mask = torch.logical_or(torch.ge(instances, 23), torch.ge(instances, 24))
    instances_masked = torch.mul(instances, face_mask)
    face_indicies = torch.nonzero(instances_masked, as_tuple=True)
    resize_diff = int((target_shape[0] - target_shape[1]) / 2)
    if torch.numel(face_indicies[0]) == 0 or torch.numel(face_indicies[1]) == 0:
        return 0
    else:
        xmin, xmax = [torch.min(face_indicies[0]).item(), torch.max(face_indicies[0]).item()]
        ymin, ymax = [torch.min(face_indicies[1]).item(), torch.max(face_indicies[1]).item()]
        cropped_face = source_image[:, xmin:xmax, ymin:ymax]
        cropped_face = cropped_face.permute((1, 2, 0)).add(1).div(2).mul(255).cpu().numpy()
        try:
            box = face_detector.detect(cropped_face)[0][0]
            if use_rendered:
                cropped_rendered_face = rendered_image[:, xmin:xmax, ymin:ymax]
                cropped_rendered_face = cropped_rendered_face.permute((1, 2, 0)).add(1).div(2).mul(255).detach().cpu().numpy()
                cropped_face = extract_face(cropped_rendered_face, box, image_size=target_shape[0])
            else:
                cropped_face = extract_face(cropped_face, box, image_size=target_shape[0])
            cropped_face = cropped_face[:, resize_diff: target_shape[0] - resize_diff, :]
            return cropped_face.cuda()
        except:
            return 0

