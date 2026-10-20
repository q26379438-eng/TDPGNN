from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import shutil

from PIL import Image
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision
import cv2
import numpy as np
import time


import os
import logging

import torch
import torch.nn as nn
import yaml

from config import cfg
from config import update_config
from core.function import get_final_preds
from utils.transforms import get_affine_transform

config_1= "/root/siton-data-zhangmingData/ZWJ/OPEC-Net-master/engineer/SPPE/src/models/w32_Simcc.yaml"

with open(config_1, 'r', encoding='utf-8') as fin:
    cfg = yaml.load(fin, Loader=yaml.FullLoader)

CROWDPOSE_KEYPOINT_INDEXES = {
    0: 'left_shoulder',
    1: 'right_shoulder',
    2: 'left_elbow',
    3: 'right_elbow',
    4: 'left_wrist',
    5: 'right_wrist',
    6: 'left_hip',
    7: 'right_hip',
    8: 'left_knee',
    9: 'right_knee',
    10: 'left_ankle',
    11: 'right_ankle',
    12: 'head',
    13: 'neck'
}

CROWDPOSE_SKELETON = [
    [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [6, 7], [6, 8], [7, 9], [8, 10], [9, 11], [12, 13], [0, 13], [1, 13], [6, 13], [7, 13]
]

CROWDPOSE_COLORS = [[255, 0, 0], [255, 85, 0], [255, 170, 0], [255, 255, 0], [170, 255, 0], [85, 255, 0], [0, 255, 0],
              [0, 255, 85], [0, 255, 170], [0, 255, 255], [0, 170, 255], [0, 85, 255], [0, 0, 255], [85, 0, 255],
              [170, 0, 255], [255, 0, 255], [255, 0, 170], [255, 0, 85]]

NUM_KPTS = 14  # Crowdpose关键点数量

CTX = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def draw_pose(keypoints, img):
    """Draw the keypoints and the skeletons."""
    assert keypoints.shape == (NUM_KPTS, 2)
    for i in range(len(CROWDPOSE_SKELETON)):
        kpt_a, kpt_b = CROWDPOSE_SKELETON[i][0], CROWDPOSE_SKELETON[i][1]
        x_a, y_a = keypoints[kpt_a][0], keypoints[kpt_a][1]
        x_b, y_b = keypoints[kpt_b][0], keypoints[kpt_b][1]
        cv2.circle(img, (int(x_a), int(y_a)), 6, CROWDPOSE_COLORS[i], -1)
        cv2.circle(img, (int(x_b), int(y_b)), 6, CROWDPOSE_COLORS[i], -1)
        cv2.line(img, (int(x_a), int(y_a)), (int(x_b), int(y_b)), CROWDPOSE_COLORS[i], 2)

def draw_bbox(box, img):
    """Draw the detected bounding box on the image."""
    cv2.rectangle(img, box[0], box[1], color=(0, 255, 0), thickness=3)

def get_person_detection_boxes(model, img, threshold=0.5):
    pred = model(img)
    pred_classes = [i for i in list(pred[0]['labels'].cpu().numpy())]  # Get the Prediction Score
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].detach().cpu().numpy())]  # Bounding boxes
    pred_score = list(pred[0]['scores'].detach().cpu().numpy())
    if not pred_score or max(pred_score) < threshold:
        return []
    # Get list of index with score greater than threshold
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold][-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_classes = pred_classes[:pred_t+1]

    person_boxes = []
    for idx, box in enumerate(pred_boxes):
        if pred_classes[idx] == 1:  # 'person' class index is 1
            person_boxes.append(box)

    return person_boxes

def get_pose_estimation_prediction(pose_model, image, center, scale):
    rotation = 0

    # pose estimation transformation
    trans = get_affine_transform(center, scale, rotation, cfg.MODEL.IMAGE_SIZE)
    model_input = cv2.warpAffine(
        image,
        trans,
        (int(cfg.MODEL.IMAGE_SIZE[0]), int(cfg.MODEL.IMAGE_SIZE[1])),
        flags=cv2.INTER_LINEAR)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # pose estimation inference
    model_input = transform(model_input).unsqueeze(0)
    # switch to evaluate mode
    pose_model.eval()
    with torch.no_grad():
        # compute output heatmap
        output = pose_model(model_input)
        preds, _ = get_final_preds(
            cfg,
            output.clone().cpu().numpy(),
            np.asarray([center]),
            np.asarray([scale]))

        return preds

def box_to_center_scale(box, model_image_width, model_image_height):
    """Convert a box to center,scale information required for pose transformation"""
    center = np.zeros((2), dtype=np.float32)

    bottom_left_corner = box[0]
    top_right_corner = box[1]
    box_width = top_right_corner[0]-bottom_left_corner[0]
    box_height = top_right_corner[1]-bottom_left_corner[1]
    bottom_left_x = bottom_left_corner[0]
    bottom_left_y = bottom_left_corner[1]
    center[0] = bottom_left_x + box_width * 0.5
    center[1] = bottom_left_y + box_height * 0.5

    aspect_ratio = model_image_width * 1.0 / model_image_height
    pixel_std = 200

    if box_width > aspect_ratio * box_height:
        box_height = box_width * 1.0 / aspect_ratio
    elif box_width < aspect_ratio * box_height:
        box_width = box_height * aspect_ratio
    scale = np.array(
        [box_width * 1.0 / pixel_std, box_height * 1.0 / pixel_std],
        dtype=np.float32)
    if center[0] != -1:
        scale = scale * 1.25

    return center, scale

def parse_args():
    parser = argparse.ArgumentParser(description='Train keypoints network')
    # general
    parser.add_argument('--cfg', type=str, default='demo/inference-config.yaml')
    parser.add_argument('--image_dir', type=str, help='Path to the Crowdpose image directory')
    parser.add_argument('--write', action='store_true')

    parser.add_argument('opts',
                        help='Modify config options using the command-line',
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # args expected by supporting codebase
    args.modelDir = ''
    args.logDir = ''
    args.dataDir = ''
    args.prevModelDir = ''
    return args

def main():
    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    args = parse_args()
    update_config(cfg, args)

    box_model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    box_model.to(CTX)
    box_model.eval()

    pose_model = eval('models.'+cfg.MODEL.NAME+'.get_pose_net')(
        cfg, is_train=False
    )

    if cfg.TEST.MODEL_FILE:
        print('=> loading model from {}'.format(cfg.TEST.MODEL_FILE))
        pose_model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE), strict=False)
    else:
        print('expected model defined in config at TEST.MODEL_FILE')

    pose_model = torch.nn.DataParallel(pose_model, device_ids=cfg.GPUS)
    pose_model.to(CTX)
    pose_model.eval()

    image_dir = args.image_dir
    if not os.path.exists(image_dir):
        print('please provide a valid --image_dir')
        return

    output_dir = 'output_CPS_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for image_name in os.listdir(image_dir):
        if image_name.endswith('.jpg') or image_name.endswith('.png'):
            image_path = os.path.join(image_dir, image_name)
            image_bgr = cv2.imread(image_path)
            image = image_bgr[:, :, [2, 1, 0]]

            input = []
            img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(img/255.).permute(2,0,1).float().to(CTX)
            input.append(img_tensor)

            # object detection box
            pred_boxes = get_person_detection_boxes(box_model, input, threshold=0.9)

            # pose estimation
            if len(pred_boxes) >= 1:
                for box in pred_boxes:
                    center, scale = box_to_center_scale(box, cfg.MODEL.IMAGE_SIZE[0], cfg.MODEL.IMAGE_SIZE[1])
                    image_pose = image.copy() if cfg.DATASET.COLOR_RGB else image_bgr.copy()
                    pose_preds = get_pose_estimation_prediction(pose_model, image_pose, center, scale)
                    if len(pose_preds) >= 1:
                        for kpt in pose_preds:
                            draw_pose(kpt, image_bgr)  # draw the poses

            output_image_path = os.path.join(output_dir, image_name)
            cv2.imwrite(output_image_path, image_bgr)
            print(f'Processed {image_name}, saved to {output_image_path}')

    print('Processing complete. All images have been saved to the output_images directory.')

if __name__ == '__main__':
    main()
