# Introduction

Object tracking involves identifying and following objects across frames in a video. The primary challenge in object tracking is to maintain the identity of objects over time despite occlusions, changes in appearance, and motion. In this GitHub repository, we explore and compare the performance of three object tracking algorithms: SORT, DeepSORT, and StrongSORT. SORT is a fast and simple tracking algorithm that doesn't use deep learning, while DeepSORT and StrongSORT are more advanced algorithms that use deep learning to improve tracking accuracy and robustness.

# General Object Tracking Process
![alt text](results/image.png)

1- **Detection**: Identify objects in each video frame using an object detection algorithm (here Yolov8).

2- **Initialization**: Assign a unique ID to each detected object to start tracking.

3- **Prediction**: Predict the object's future position using a model like the Kalman filter.

4- **Data Association**: Match detected objects in new frames with existing predicted tracks to maintain their identity using a model like Hungarian algorithm.

5- **Update**: Update each tracked object's state based on new information using Kalman filter.

6- **Track Management**: Start new tracks for new detections, update matched tracks, and delete or age out missing tracks.

# DeepSORT Object Tracking Process
![alt text](results/image1.png)

1- **Detection**: Identify objects in each video frame using an object detection algorithm (here Yolov8).

2- **Initialization**: Assign a unique ID to each detected object to start tracking.

3- **Prediction**: Predict the object's future position using a model like the Kalman filter.

3- **Appearance Feature Extraction**: For each detected object, a deep appearance descriptor (typically a pretrained CNN) is used to extract appearance features.

 3-1 **Cosine Similarity Calculation**:The cosine similarity (or cosine distance) between the features of new detections and the stored features in the feature bank is calculated. This similarity score is used to measure the appearance-based matching cost.

4- **Data Association**: Match detected objects in new frames with existing predicted tracks to maintain their identity using a model like Hungarian algorithm.

 4-1 **Both appearance and motion information** are used to associate new detections with existing tracklets.
 The matching cost matrix is created using the appearance-based matching cost (cosine similarity) and motion-based matching cost (IOU)

5- **Update**: Update each tracked object's state based on new information using Kalman filter.
 .

6- **Track Management**: Start new tracks for new detections, update matched tracks, and delete or age out missing tracks.

