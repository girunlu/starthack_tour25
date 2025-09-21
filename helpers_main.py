import cv2
import torch
import numpy as np
from sklearn.cluster import DBSCAN

def detect_moving_clusters(frame1, frame2, previous_points=None, 
                            motion_threshold=1.0, dbscan_eps=30, dbscan_min_samples=3):
    """
    Analyze two frames and return bounding boxes for moving object clusters.
    
    Args:
        frame1: Previous frame (numpy array)
        frame2: Current frame (numpy array) 
        previous_points: Feature points from previous frame (optional)
        motion_threshold: Minimum pixel movement to consider as motion
        dbscan_eps: DBSCAN clustering distance parameter
        dbscan_min_samples: Minimum samples for DBSCAN cluster
    
    Returns:
        tuple: (bounding_boxes, new_feature_points)
        - bounding_boxes: List of (x1, y1, x2, y2) tuples
        - new_feature_points: Feature points for next iteration
    """
    
    # Convert to grayscale if needed
    if len(frame1.shape) == 3:
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = frame1
        
    if len(frame2.shape) == 3:
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = frame2
    
    # Get feature points if not provided
    if previous_points is None:
        previous_points = cv2.goodFeaturesToTrack(
            gray1, maxCorners=500, qualityLevel=0.01, minDistance=5
        )
    
    if previous_points is None or len(previous_points) == 0:
        return [], None
    
    # Optical flow parameters
    lk_params = dict(
        winSize=(15, 15),
        maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
    )
    
    # Calculate optical flow
    new_points, status, error = cv2.calcOpticalFlowPyrLK(gray1, gray2, previous_points, None, **lk_params)
    
    if new_points is None:
        return [], None
    
    # Filter good points
    good_new = new_points[status == 1]
    good_old = previous_points[status == 1]
    
    # Calculate motion magnitude
    motion = np.linalg.norm(good_new - good_old, axis=1)
    moving_points = good_new[motion > motion_threshold]
    
    bounding_boxes = []
    
    if len(moving_points) > 0:
        # Cluster moving points
        clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(moving_points)
        labels = clustering.labels_
        
        # Create bounding box for each cluster
        for label in set(labels):
            if label == -1:  # Skip noise points
                continue
                
            cluster_points = moving_points[labels == label]
            
            # Get bounding box coordinates
            x_min, y_min = cluster_points.min(axis=0)
            x_max, y_max = cluster_points.max(axis=0)
            
            # Expand box to square with padding
            width = x_max - x_min
            height = y_max - y_min
            side = int(max(width, height) * 1.25)  # 25% padding
            
            # Center the square box
            center_x = int((x_min + x_max) / 2)
            center_y = int((y_min + y_max) / 2)
            
            x1 = max(0, center_x - side // 2)
            y1 = max(0, center_y - side // 2)
            x2 = min(frame2.shape[1], center_x + side // 2)
            y2 = min(frame2.shape[0], center_y + side // 2)
            
            bounding_boxes.append((x1, y1, x2, y2))
    
    # Return new feature points for next iteration
    new_feature_points = good_new.reshape(-1, 1, 2) if len(good_new) > 0 else None
    
    return bounding_boxes, new_feature_points


def detect_objects_yolo(frame, model, img_size=640):
    """
    Detect objects in a single frame using YOLOv5.
    
    Args:
        frame: OpenCV frame array (BGR format)
        model: Pre-loaded YOLOv5 model (initialized outside this function)
        confidence_threshold: Minimum confidence for detection
        img_size: Input size for YOLO model
        
    Returns:
        List of detection tuples: [(x1, y1, x2, y2, confidence, class_id, class_name), ...]
        Returns empty list [] if no detections found.
        
        Each detection tuple contains:
        - x1, y1: Top-left corner coordinates
        - x2, y2: Bottom-right corner coordinates  
        - confidence: Detection confidence score (0.0 to 1.0)
        - class_id: Integer class ID
        - class_name: String class name
    """
    if frame is None or frame.size == 0:
        return []

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = model(frame_rgb, size=img_size)
    detections = []
    pred = results.pred[0]  # predictions for first image
    if len(pred) == 0:
        return []

    pred = pred.cpu().numpy()
    for detection in pred:
        x1, y1, x2, y2, conf, class_id = detection[:6]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        class_id = int(class_id)
        class_name = model.names[1]
        detections.append((x1, y1, x2, y2, float(conf), class_id, class_name))
    
    return detections


def estimate_depth(image, model, normalize=True):
    """
    Estimate depth from a single image using DepthAnythingV2.
    
    Args:
        image: OpenCV image array (BGR format)
        model: Pre-loaded DepthAnythingV2 model (initialized outside this function)
        normalize: If True, normalize depth values to [0, 1] range
        
    Returns:
        numpy.ndarray: Depth matrix (HxW)
        - If normalize=True: values in range [0, 1] where 0=closest, 1=farthest
        - If normalize=False: raw depth values (inverted from model output)
        
        Returns None if image is invalid.
    """
    if image is None or image.size == 0:
        return None

    depth = model.infer_image(image)
    depth = abs(depth - depth.max())
    
    if normalize:
        depth_min = depth.min()
        depth_max = depth.max()
        
        if depth_max - depth_min > 0:
            depth = (depth - depth_min) / (depth_max - depth_min)
        else:
            depth = np.zeros_like(depth)
    
    return depth


def yolo_drone_3d(box1, box2, timestamp, 
                                    drone_size_m=0.5, 
                                    focal_length_mm=24, sensor_width_mm=36, 
                                    image_width_px=1920, image_height_px=1600):
    """
    Calculate 3D coordinates and speed of a detected object from YOLO boxes.

    Args:
        box1: [x1, y1, x2, y2] of first detection (pixels)
        box2: [x1, y1, x2, y2] of second detection (pixels)
        timestamp: time difference between frames (seconds)
        drone_size_m: real width of the object (meters)
        focal_length_mm: camera focal length (mm)
        sensor_width_mm: sensor width (mm)
        image_width_px: camera resolution width (pixels)
        image_height_px: camera resolution height (pixels)

    Returns:
        dict with:
            - position1_3d_m: (x, y, z) in meters at frame 1
            - position2_3d_m: (x, y, z) in meters at frame 2
            - speed_m_s: speed in meters/second
    """

    def box_center_and_width(box):
        print('BOX', box)
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        return center_x, center_y, width

    # Focal length in pixels
    f_px = focal_length_mm * (image_width_px / sensor_width_mm)

    # Frame 1
    cx1, cy1, w1 = box_center_and_width(box1)
    z1 = f_px * drone_size_m / w1
    x1_m = (cx1 - image_width_px/2) * z1 / f_px
    y1_m = (cy1 - image_height_px/2) * z1 / f_px

    # Frame 2
    cx2, cy2, w2 = box_center_and_width(box2)
    z2 = f_px * drone_size_m / w2
    x2_m = (cx2 - image_width_px/2) * z2 / f_px
    y2_m = (cy2 - image_height_px/2) * z2 / f_px

    # Speed
    dx = x2_m - x1_m
    dy = y2_m - y1_m
    dz = z2 - z1
    distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    speed = distance_3d / timestamp

    return {
        "position1_3d_m": (x1_m, y1_m, z1),
        "position2_3d_m": (x2_m, y2_m, z2),
        "speed_m_s": speed
    }


def yolo_plane_3d(box1, box2, timestamp, 
                    plane_length_m=30.0, plane_wingspan_m=28.0,
                    focal_length_mm=24, sensor_width_mm=36,
                    image_width_px=1920, image_height_px=1600):
    """
    Calculate 3D coordinates and speed of a plane from YOLO boxes.
    Uses longer and shorter sides of bounding box for better scaling.
    
    Args:
        box1: [x1, y1, x2, y2] of first detection (pixels)
        box2: [x1, y1, x2, y2] of second detection (pixels)
        timestamp: time difference between frames (seconds)
        plane_length_m: physical length of plane (meters)
        plane_wingspan_m: physical wingspan of plane (meters)
        focal_length_mm: camera focal length (mm)
        sensor_width_mm: sensor width (mm)
        image_width_px: camera width in pixels
        image_height_px: camera height in pixels
    
    Returns:
        dict with:
            - position1_3d_m: (x, y, z) at frame 1
            - position2_3d_m: (x, y, z) at frame 2
            - speed_m_s: 3D speed in meters/second
    """

    def box_center_and_sides(box):
        x1, y1, x2, y2 = box
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        width = x2 - x1
        height = y2 - y1
        longer = max(width, height)
        shorter = min(width, height)
        return center_x, center_y, longer, shorter

    # Focal length in pixels
    f_px = focal_length_mm * (image_width_px / sensor_width_mm)

    # Frame 1
    cx1, cy1, long1, short1 = box_center_and_sides(box1)
    # Scale factor: use the longer side to estimate distance
    z1_long = f_px * plane_length_m / long1
    z1_short = f_px * plane_wingspan_m / short1
    z1 = (z1_long + z1_short) / 2  # average for robustness
    x1_m = (cx1 - image_width_px/2) * z1 / f_px
    y1_m = (cy1 - image_height_px/2) * z1 / f_px

    # Frame 2
    cx2, cy2, long2, short2 = box_center_and_sides(box2)
    z2_long = f_px * plane_length_m / long2
    z2_short = f_px * plane_wingspan_m / short2
    z2 = (z2_long + z2_short) / 2
    x2_m = (cx2 - image_width_px/2) * z2 / f_px
    y2_m = (cy2 - image_height_px/2) * z2 / f_px

    # Speed calculation
    dx = x2_m - x1_m
    dy = y2_m - y1_m
    dz = z2 - z1
    distance_3d = np.sqrt(dx**2 + dy**2 + dz**2)
    speed = distance_3d / timestamp

    return {
        "position1_3d_m": (x1_m, y1_m, z1),
        "position2_3d_m": (x2_m, y2_m, z2),
        "speed_m_s": speed
    }


def vis_yolo(frame, yolo_detections):
    """Draw YOLO detection boxes on frame."""
    for detection in yolo_detections:
        x1, y1, x2, y2, conf, class_id, class_name = detection
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        cv2.putText(frame, f"{class_name}: {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)


def vis_motion(frame, motion_boxes):
    """Draw motion detection boxes on frame."""
    for i, (x1, y1, x2, y2) in enumerate(motion_boxes):
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"Motion {i+1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


def send_data(sequence_data):
    """
    Placeholder function for storing processed sequence data.
    To be implemented later depending on the service.
    
    Args:
        sequence_data: Dict containing all processed sequence information
    """
    # This function will be implemented
    # It receives processed sequence data for storage/further processing
    pass


def process_alignment_sequence(alignment_sequence, fps):
    """
    Process a complete alignment sequence (start to end).
    
    Args:
        alignment_sequence: List of aligned detection data
        fps: Video frame rate
    """
    if len(alignment_sequence) < 2:
        return
    
    # Get start and end detections
    start_detection = alignment_sequence[0]
    end_detection = alignment_sequence[-1]
    
    # Extract class information
    class_id = start_detection['class_id']
    class_name = start_detection['class_name']
    
    print(f"Processing sequence: {class_name} (class {class_id})")
    print(f"Duration: {end_detection['timestamp'] - start_detection['timestamp']:.2f}s")
    print(f"Frames: {len(alignment_sequence)}")
    
    # Check if class is drone (0) or plane (1)
    if class_id == 0:
        # Run speed and location analysis
        speed_data = yolo_drone_3d(start_detection, end_detection, fps)
        sequence_data = {
            'class_id': class_id,
            'class_name': class_name,
            'start_detection': start_detection,
            'end_detection': end_detection,
            'sequence_length': len(alignment_sequence),
            'duration': end_detection['timestamp'] - start_detection['timestamp'],
            'speed_data': speed_data,
            'all_detections': alignment_sequence
        }
        # Call the arbitrary feed_store function (to be implemented)
        send_data(sequence_data)
        return sequence_data
        
    if class_id == 1:
        speed_data = yolo_plane_3d(start_detection, end_detection, fps)
        sequence_data = {
            'class_id': class_id,
            'class_name': class_name,
            'start_detection': start_detection,
            'end_detection': end_detection,
            'sequence_length': len(alignment_sequence),
            'duration': end_detection['timestamp'] - start_detection['timestamp'],
            'speed_data': speed_data,
            'all_detections': alignment_sequence
        }
        # Call the arbitrary feed_store function (to be implemented)
        send_data(sequence_data)
        return sequence_data


def check_alignment(motion_boxes, yolo_detections):
    """
    Check if any motion cluster center is inside a YOLO detection box.
    
    Args:
        motion_boxes: List of motion bounding boxes [(x1,y1,x2,y2), ...]
        yolo_detections: List of YOLO detections [(x1,y1,x2,y2,conf,cls_id,cls_name), ...]
    
    Returns:
        bool: True if alignment found
    """
    for mx1, my1, mx2, my2 in motion_boxes:
        # Calculate motion cluster center
        motion_center_x = (mx1 + mx2) // 2
        motion_center_y = (my1 + my2) // 2
        
        for detection in yolo_detections:
            yx1, yy1, yx2, yy2 = detection[:4]
            
            # Check if motion center is inside YOLO box
            if yx1 <= motion_center_x <= yx2 and yy1 <= motion_center_y <= yy2:
                return True
    return False

