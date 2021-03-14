import os
import h5py
import numpy as np
from shutil import copyfile, rmtree
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import torch.nn as nn
import torchvision.models as models
from losses import SaveOutput

def load_h5_file(path):
    with h5py.File(path, mode="r") as h5_file:
        frame = h5_file["frame"][:].astype(np.float32)
        instances = h5_file["i"][:].astype(np.float32)
        texture = h5_file["texture"][:].astype(np.float32)
        uv = h5_file["uv"][:].astype(np.float32)
        joints = h5_file["joints"][:].astype(np.float32)
    return frame, instances, texture, uv, joints


def parse_deepfashion():
    deepfashion_path = '/datasets/deepfashion/processed'
    filtered_path = '/home/pkowaleczko/datasets/deepfashion/deepfashion_filtered/'
    if os.path.exists(filtered_path):
        print('Folder deleted')
        rmtree(filtered_path)

    ids = set()
    for path, subdirs, files in os.walk(deepfashion_path):
        for name in files:
            ids.add(path)

    filepath = ''
    for id in ids:
        qualified_files = []
        for path, subdirs, files in os.walk(id):
            for name in files:
                filepath = os.path.join(path, name)
                frame, instances, texture, uv, joints = load_h5_file(filepath)
                detected_joints = 0
                for joint in joints:
                    if not np.isnan(joint).any():
                        detected_joints += 1
                if detected_joints > 10:
                    qualified_files.append(filepath)
        if len(qualified_files) > 1:
            person_ids = []
            for file in qualified_files:
                person_ids.append(os.path.basename(file).split('_')[0])
            ids_to_remove = []
            for pid in set(person_ids):
                if person_ids.count(pid) == 1:
                    ids_to_remove.append(pid)
            final_files = []
            for file in qualified_files:
                if os.path.basename(file).split('_')[0] in ids_to_remove:
                    pass
                else:
                    final_files.append(file)
            for file in final_files:
                dst_path = filtered_path + '/'.join(file.split('/')[-4:])
                os.makedirs(os.path.dirname(dst_path), exist_ok=True)
                copyfile(file, dst_path)

def get_face_embeddings(dataset_path):
    mtcnn = MTCNN(image_size=160, device='cuda', select_largest=True)
    face_rec = InceptionResnetV1(pretrained='vggface2').eval().cuda()
    vgg19 = models.vgg19(pretrained=True).eval().cuda()
    hook = SaveOutput()
    hook_handles = []
    with torch.no_grad():
        for layer in vgg19.modules():
            if isinstance(layer, nn.ReLU):
                handle = layer.register_forward_hook(hook)
                hook_handles.append(handle)
        for idx, (path, subdirs, files) in enumerate(os.walk(dataset_path)):
            embeddings = []
            for name in files:
                if name.endswith('h5'):
                    f, i, t, u, j = load_h5_file(os.path.join(path, name))
                    f_features = torch.from_numpy(f).permute((2, 0 , 1))
                    print(f_features)
                    vgg19(f_features)
                    ground_truth_activations = hook.outputs
                    print(ground_truth_activations)
                    hook.clear()
                    try:
                        face = torch.unsqueeze(mtcnn(f), 0).cuda()
                        embedding = face_rec(face)
                        embeddings.append(embedding)
                    except Exception as e:
                        embedding = torch.zeros((1, 512))


get_face_embeddings('/home/pkowaleczko/datasets/deepfashion/deepfashion_filtered')
# parse_deepfashion()