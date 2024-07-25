import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import copy
import glob
import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO
import torch
import os

# Import StrongSORT++ tracker
# pip install gdown
from strong_sort import StrongSORT
from strong_sort.utils.parser import get_config
from track_utils import strongsort_instances, img_preprocess, rescale_bbox, process_results, get_boxes_result, inference_with_nms

def verify_weights_file(file_path):
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"Weight file not found: {file_path}")
    try:
        checkpoint = torch.load(file_path, map_location='cpu')
        print("Weight file loaded successfully.")
    except Exception as e:
        raise IOError(f"Error loading weight file: {file_path}\n{str(e)}")

# def load_weights(model, file_path):
#     try:
#         state_dict = torch.load(file_path, map_location='cpu')
#         model.load_state_dict(state_dict,strict=False)
#         print("Model weights loaded successfully.")
#     except Exception as e:
#         raise IOError(f"Error loading model weights from file: {file_path}\n{str(e)}")

def visualize_images(input_images):
    fig = plt.figure(figsize=(15, 7.5))
    for i in range(len(input_images)):
        fig.add_subplot(1, len(input_images), i+1)
        plt.imshow(input_images[i])
    plt.show()

# images_files = sorted(glob.glob("../tracking_course/data/*.png"))
images_files = sorted(glob.glob("../tracking_course/test_videos/test_video1/extracted_frames/*.png"))

images = []

# Example indexes to test on:
# 100 — Same number of detections
# 300 — 2 Lost Tracks
# 782 — 1 New Detection
# 654 — Edge Case

# index = 782
index = 10
for img in images_files[index:index+2]:
    images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))

visualize_images(images)

### YOLOv8 Inference ###########################################

def get_results(results):
    xywhs = []
    confs = []
    clss = []
    for result in results[0]:
        bbox = result.boxes.xywh.cpu().numpy()
        confidence = result.boxes.conf.cpu().numpy()
        class_id = result.boxes.cls.cpu().numpy().astype(int)        
        xywhs.append(bbox[0])
        confs.append(confidence[0])
        clss.append(class_id[0])
        
    return np.array(xywhs), np.array(confs), np.array(clss)

def draw_bounding_boxes(img, bboxes, ids, frame_rate):
    for bbox, id_ in zip(bboxes, ids):
        cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
        cv2.putText(img, "ID: " + str(id_), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    # Write the frame rate on the image
    cv2.putText(img, f"FPS: {frame_rate}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return img

# Load the YOLOv8 model
model = YOLO('yolov8l.pt')
yolo_images = []
pics = copy.deepcopy(images)
for img in pics:
    results = model.predict(img, classes=2)
    annotated_frame = results[0].plot()
    yolo_images.append(annotated_frame)

visualize_images(yolo_images)

# Initialize StrongSORT++ Tracker
config_strongsort = 'strong_sort/configs/strong_sort.yaml'
# strong_sort_weights = './osnet_x0_25_msmt17.pt'
strong_sort_weights = 'osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth'
cfg = get_config()
cfg.merge_from_file(config_strongsort)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

nr_sources = 1  # Adjust if you have multiple sources
strongsort_list = strongsort_instances(nr_sources, strong_sort_weights, device, cfg)

out_imgs = []
j = 0
fig = plt.figure(figsize=(15, 7.5))
for i in range(len(images)):
    results = model.predict(images[i], classes=[2,7])
    xywhs, confs, clss = get_results(results)
    if len(xywhs) == 0:
        xywhs = np.empty((0, 4))
        confs = np.empty((0,))
        clss = np.empty((0,))
    # update output format df
    res = strongsort_list[0].update(xywhs, confs, clss, images[i])
    # Ensure res is a NumPy array
    if isinstance(res, list):
        res = np.array(res)

    # Check if res is empty
    if res.size > 0:
        boxes_track = res[:, :4]
        boxes_ids = res[:, 4].astype(int)

        # Visualize results
        frame = draw_bounding_boxes(images[i], boxes_track, boxes_ids, frame_rate=30)
        out_imgs.append(frame)
        fig.add_subplot(1, len(images), j+1)
        plt.imshow(out_imgs[j])
        j +=1
    else:
        print("Warning: No tracking results returned.")
    

plt.show()

# Video Processing
video_images = sorted(glob.glob("../tracking_course/test_videos/test_video1/extracted_frames/*.png"))
frame_height, frame_width, _ = images[0].shape
out = cv2.VideoWriter('my_strongsort_v8_v2.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 15, (frame_width, frame_height))

for img_path in tqdm(video_images):
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    results = model.predict(img, classes=[2,7])
    xywhs, confs, clss = get_results(results)
    if len(xywhs) == 0:
        xywhs = np.empty((0, 4))
        confs = np.empty((0,))
        clss = np.empty((0,))

    res = strongsort_list[0].update(xywhs, confs, clss, img)
    boxes_track = res[:, :4]
    boxes_ids = res[:, 4].astype(int)

    frame = draw_bounding_boxes(img, boxes_track, boxes_ids, frame_rate=15)  # Assuming 15 FPS for video
    out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

out.release()