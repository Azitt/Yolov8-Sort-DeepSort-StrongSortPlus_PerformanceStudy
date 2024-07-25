import argparse
import cv2
import numpy as np
from ultralytics import YOLO
from sort import Sort
from deep_sort_realtime.deepsort_tracker import DeepSort
from track_utils import strongsort_instances
from strong_sort.utils.parser import get_config
import copy
import glob
import torch
import matplotlib.pyplot as plt

class Tracker:
    def __init__(self, tracker_type='strongsort', config_strongsort=None, strong_sort_weights=None):
        self.tracker_type = tracker_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        if tracker_type == 'sort':
            self.tracker = Sort(max_age=30, min_hits=1, iou_threshold=0.3)
        elif tracker_type == 'deepsort':
            self.tracker = DeepSort(max_age=30, n_init=10, nms_max_overlap=1.0, max_iou_distance=0.7, max_cosine_distance=0.2)
        elif tracker_type == 'strongsort':
            cfg = get_config()
            cfg.merge_from_file(config_strongsort)
            self.nr_sources = 1 # Adjust if you have multiple sources
            self.tracker = nr_sources = 1  
            self.tracker = strongsort_instances(nr_sources, strong_sort_weights, self.device, cfg)

    def update(self, detections, frame):
        if self.tracker_type == 'sort':
            return self.tracker.update(detections)
        elif self.tracker_type == 'deepsort':
            return self.tracker.update_tracks(detections, frame=frame)
        elif self.tracker_type == 'strongsort':
            xywhs, confs, clss = detections
            if len(xywhs) == 0:
                xywhs = np.empty((0, 4))
                confs = np.empty((0,))
                clss = np.empty((0,))
            return self.tracker[0].update(xywhs, confs, clss, frame)
    
    def draw_bounding_boxes(self, img, bboxes, ids, frame_rate):
        for bbox, id_ in zip(bboxes, ids):
            cv2.rectangle(img, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 255), 2)
            cv2.putText(img, "ID: " + str(id_), (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {frame_rate}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        return img

def get_results_strong(results):
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

def get_results_deep(results):
        
        xyxys = []
        confidences = []
        class_ids = []
        detections_list = []
        
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            #if class_id == 0:
                
            bbox = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            
            merged_detection = ([bbox[0][0], bbox[0][1], bbox[0][2]-bbox[0][0], bbox[0][3]-bbox[0][1]], confidence, class_id)
            
            detections_list.append(merged_detection)
            xyxys.append(bbox)
            confidences.append(confidence)
            class_ids.append(class_id)
            
    
        return detections_list

def get_results(results):
        
        xyxys = []
        confidences = []
        class_ids = []
        detections_list = []
        
        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            
            
            #if class_id == 0:
                
            bbox = result.boxes.xyxy.cpu().numpy()
            confidence = result.boxes.conf.cpu().numpy()
            
            
            merged_detection = [bbox[0][0], bbox[0][1], bbox[0][2], bbox[0][3], confidence[0]]
            
            
            detections_list.append(merged_detection)
            xyxys.append(bbox)
            confidences.append(confidence)
            class_ids.append(class_id)
            
    
        return np.array(detections_list)

def visualize_images(input_images):
    fig = plt.figure(figsize=(15, 7.5))
    for i in range(len(input_images)):
        fig.add_subplot(1, len(input_images), i+1)
        plt.imshow(input_images[i])
    plt.show()

def main():
    parser = argparse.ArgumentParser(description="Object Tracking using YOLO and SORT/DeepSORT/StrongSORT++")
    parser.add_argument('--tracker', type=str, default='strongsort', choices=['sort', 'deepsort', 'strongsort'],
                        help='Tracker type: sort, deepsort, or strongsort')
    args = parser.parse_args()

    # Load the YOLOv8 model
    model = YOLO('yolov8l.pt')

    # Initialize the tracker
    config_strongsort = 'strong_sort/configs/strong_sort.yaml'
    strong_sort_weights = 'osnet_x0_25_msmt17_256x128_amsgrad_ep180_stp80_lr0.003_b128_fb10_softmax_labelsmooth_flip.pth'
    tracker = Tracker(args.tracker, config_strongsort, strong_sort_weights)

    # Load images
    images_files = sorted(glob.glob("../data/*.png"))
    images = []
    index = 10
    for img in images_files[index:index+2]:
        images.append(cv2.cvtColor(cv2.imread(img), cv2.COLOR_BGR2RGB))
    visualize_images(images)

    # Process images
    out_imgs = []
    for img in images:
        results = model.predict(img)
        
        if args.tracker == 'sort':
         detections = get_results(results)
        elif args.tracker == 'deepsort':           
         detections = get_results_deep(results)
        else:
         detections = get_results_strong(results)   
         
        tracks = tracker.update(detections, img)

        if len(tracks) > 0:
            if args.tracker == 'sort':
                boxes_track = tracks[:,:-1]
                boxes_ids = tracks[:,-1].astype(int)
            elif args.tracker == 'strongsort':
               boxes_track = tracks[:, :4]
               boxes_ids = tracks[:, 4].astype(int)    
            else:
                bboxes = []
                track_ids = []
                for track in tracks:
                    if not track.is_confirmed():
                        continue
                    bboxes.append(track.to_ltrb())
                    track_ids.append(track.track_id)
                boxes_track = np.array(bboxes)
                boxes_ids = np.array(track_ids)
            
            frame = tracker.draw_bounding_boxes(img, boxes_track, boxes_ids, frame_rate=30)
            out_imgs.append(frame)

    visualize_images(out_imgs)

if __name__ == "__main__":
    main()
