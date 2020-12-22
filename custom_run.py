import numpy as np

from common.arguments import parse_args
import torch

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.generators import ChunkedGenerator, UnchunkedGenerator
from time import time
from common.utils import deterministic_random
from common.custom_dataset import CustomDataset  

def get_model(checkpoint_path, n_keypoints=17, n_coords=2):
    filter_widths = [3, 3, 3, 3, 3]
    model = TemporalModel(n_keypoints, n_coords, n_keypoints,
                              filter_widths=filter_widths, causal=False, dropout=0.25, channels=1024,
                              dense=False)

    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint['model_pos'])

    if torch.cuda.is_available():
        model = model.cuda()

    return model

def predict_from_dataset(model, dataset_path, checkpoint_path='pretrained_h36m_detectron_coco.bin'):

    dataset = CustomDataset(dataset_path)

    subject = list(dataset._data.keys())[0]
    action = list(dataset._data[subject].keys())[0]

    # Load keypoints
    keypoints = np.load(dataset_path, allow_pickle=True)
    keypoints_metadata = keypoints['metadata'].item()
    keypoints_symmetry = keypoints_metadata['keypoints_symmetry']
    kps_left, kps_right = list(keypoints_symmetry[0]), list(keypoints_symmetry[1])
    joints_left, joints_right = list(dataset.skeleton().joints_left()), list(dataset.skeleton().joints_right())
    keypoints = keypoints['positions_2d'].item()

    # Normalize camera frame
    for cam_idx, kps in enumerate(keypoints[subject][action]):
        cam = dataset.cameras()[subject][cam_idx]
        kps[..., :2] = normalize_screen_coordinates(kps[..., :2], w=cam['res_w'], h=cam['res_h'])
        keypoints[subject][action][cam_idx] = kps

    receptive_field = model.receptive_field()
    pad = (receptive_field - 1) // 2 # Padding on each side
    causal_shift = 0

    # Make predictions
    input_keypoints = keypoints[subject][action][0].copy()

    generator = UnchunkedGenerator(None, None, [input_keypoints],
                            pad=pad, causal_shift=causal_shift, augment=True,
                            kps_left=kps_left, kps_right=kps_right, joints_left=joints_left, joints_right=joints_right)

    _, _, batch_2d = next(generator.next_epoch())
    inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

    with torch.no_grad():
        model.eval()

        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()

        predicted_3d_pos = model(inputs_2d)

        # Test-time augmentation (if enabled)
        if generator.augment_enabled():
            # Undo flipping and take average with non-flipped version
            predicted_3d_pos[1, :, :, 0] *= -1
            predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
            predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)
            
        prediction = predicted_3d_pos.squeeze(0).cpu().numpy()

        return prediction

if __name__ == "__main__":
    model = get_model('/home/david/Documents/VideoPose3D/checkpoint/pretrained_h36m_detectron_coco.bin')  
    predictions = predict_from_dataset(model, 'data/data_2d_custom_federer.npz')
    print(predictions)
