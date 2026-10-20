import cv2
import numpy as np
import matplotlib.pyplot as plt
from mmpose.apis import init_pose_model, inference_top_down_pose_model, vis_pose_result
from mmdet.apis import inference_detector, init_detector
import os

# 初始化模型
config_file_1 = '/root/siton-data-zhangmingData/ZWJ/OPEC-Net-master/engineer/SPPE/src/models/pose_hrnet_Simcc.py'
checkpoint_file_1 = '/root/siton-data-zhangmingData/ZWJ/OPEC-Net-master/weights/hrnet/pose_hrnet_w32_Simcc.pth'
model_1 = init_pose_model(config_file_1, checkpoint_file_1)

config_file_2 = '/root/siton-data-zhangmingData/ZWJ/OPEC-Net-master/engineer/SPPE/src/models/layers/pose_hrnet_Simcc_GCN.py'
checkpoint_file_2 = '/root/siton-data-zhangmingData/ZWJ/OPEC-Net-master/weights/hrnet/pose_hrnet_Simcc_GCN.pth'
model_2 = init_pose_model(config_file_2, checkpoint_file_2)

# 数据集路径
dataset_dir = '/root/siton-data-zhangmingData/ZWJ/OPEC-Net-master/crowdpose/images'
output_dir = 'outputduibi'
os.makedirs(output_dir, exist_ok=True)

# 读取图片
image_files = [f for f in os.listdir(dataset_dir) if f.endswith('.jpg')]

for image_file in image_files:
    img_path = os.path.join(dataset_dir, image_file)
    img = cv2.imread(img_path)

    # 模型1推理
    result_1 = inference_top_down_pose_model(model_1, img)
    vis_img_1 = vis_pose_result(model_1, img, result_1, dataset='CrowdPose')

    # 模型2推理
    result_2 = inference_top_down_pose_model(model_2, img)
    vis_img_2 = vis_pose_result(model_2, img, result_2, dataset='CrowdPose')

    # 拼接结果
    combined_img = np.concatenate((vis_img_1, vis_img_2), axis=1)

    # 保存结果
    output_path = os.path.join(output_dir, f'output_{image_file}')
    cv2.imwrite(output_path, combined_img)
    plt.imshow(cv2.cvtColor(combined_img, cv2.COLOR_BGR2RGB))
    plt.show()

    print(f"Processed {image_file}")

print("Processing complete.")
