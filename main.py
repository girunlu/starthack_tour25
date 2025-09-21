import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import cv2
import torch
import numpy as np
import tkinter as tk
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from depth_anything_v2.depth_anything_v2.dpt import DepthAnythingV2
from helpers_main import detect_moving_clusters, detect_objects_yolo, \
    estimate_depth, vis_yolo, vis_motion, check_alignment, process_alignment_sequence

DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'
model_configs = {'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]}}
encoder = 'vitb'

root = tk.Tk()
screen_w, screen_h = root.winfo_screenwidth(), root.winfo_screenheight()
root.destroy()

yolo_model = torch.hub.load('./yolov5', 'custom',
                        path='./yolov5/runs/train/my_experiment11/weights/best.pt', # 8 11 14
                        source='local')

# Your initialization code
depth_model = DepthAnythingV2(**model_configs[encoder])
depth_model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
depth_model = depth_model.to(DEVICE).eval()

def process_video(video_path, yolo_model, motion_params=None, alignment_params=None, save_trajectory=True):
    """
    Process video for motion detection, YOLO detection, and alignment tracking.
    
    Args:
        video_path: Path to video file
        yolo_model: Pre-loaded YOLO model
        motion_params: Dict with motion detection parameters
        alignment_params: Dict with alignment parameters
        enable_live_plot: Boolean to enable live plotting on world map
    
    Returns:
        List of processed sequences with timing and classification data
    """
    
    # Default parameters
    if motion_params is None:
        motion_params = {
            'motion_threshold': 1.0,
            'dbscan_eps': 30,
            'dbscan_min_samples': 3
        }
    
    if alignment_params is None:
        alignment_params = {
            'max_unaligned_frames': 6,  # frames to ignore without alignment
            'max_sequence_duration': 5.0,  # seconds
            'yolo_confidence': 0.25
        }
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Calculate frame buffer size (5 seconds window)
    buffer_size = int(fps * 5)
    frame_buffer = deque(maxlen=buffer_size)
    
    # Initialize tracking variables
    previous_points = None
    alignment_sequence = []  # stores aligned detections
    processed_sequences = []
    
    # Alignment tracking
    is_aligned = False
    alignment_start_time = None
    unaligned_count = 0
    
    frame_count = 0
    
    print(f"Processing video: {video_path}")
    print(f"FPS: {fps}, Total frames: {total_frames}, Buffer size: {buffer_size}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        current_time = frame_count / fps
        
        # Add frame to buffer
        frame_buffer.append(frame.copy())
        
        # Need at least 2 frames for motion detection
        if len(frame_buffer) < 2:
            continue
        
        # Run motion detection on current and previous frame
        prev_frame = frame_buffer[-2]
        current_frame = frame_buffer[-1]
        
        motion_boxes, previous_points = detect_moving_clusters(
            prev_frame, current_frame, previous_points,
            motion_threshold=motion_params['motion_threshold'],
            dbscan_eps=motion_params['dbscan_eps'],
            dbscan_min_samples=motion_params['dbscan_min_samples']
        )
        
        # Check if motion detected
        motion_detected = len(motion_boxes) > 0
        alignment_found = False
        
        if motion_detected:
            vis_motion(current_frame, motion_boxes)
            # Run YOLO on current frame
            yolo_detections = detect_objects_yolo(
                current_frame, yolo_model) 
                #confidence_threshold=alignment_params['yolo_confidence'])
            if yolo_detections:
                vis_yolo(current_frame, yolo_detections)
                # Check alignment between motion clusters and YOLO boxes
                alignment_found = check_alignment(motion_boxes, yolo_detections)
                
                if alignment_found:
                    # Store aligned detection
                    detection_data = {
                        'frame': current_frame.copy(),
                        'frame_number': frame_count,
                        'timestamp': current_time,
                        'motion_boxes': motion_boxes,
                        'yolo_detections': yolo_detections,
                        'yolo_box': yolo_detections[0][:4],  # assuming single detection
                        'class_id': yolo_detections[0][5],
                        'class_name': yolo_detections[0][6],
                        'confidence': yolo_detections[0][4]
                    }
                    alignment_sequence.append(detection_data)
            
            
            h, w = current_frame.shape[:2]
            # scale factor to fit inside screen
            scale = min(screen_w / w, screen_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(current_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
                    
            
            cv2.imshow("Detection Results", resized)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        # Handle alignment state changes
        if alignment_found:
            unaligned_count = 0
            if not is_aligned:
                alignment_start_time = current_time
                print(f"Alignment detected at {current_time:.2f}s")
        else:
            if is_aligned:
                unaligned_count += 1
                
                # Check if alignment should end
                time_exceeded = (current_time - alignment_start_time) >= alignment_params['max_sequence_duration']
                frames_exceeded = unaligned_count > alignment_params['max_unaligned_frames']
                
                if time_exceeded or frames_exceeded:
                    # End alignment sequence and process it
                    if len(alignment_sequence) >= 2:
                        sequence_data = process_alignment_sequence(alignment_sequence, fps)
                        processed_sequences.append(alignment_sequence.copy())

                    # Reset alignment tracking
                    is_aligned = False
                    alignment_sequence.clear()
                    unaligned_count = 0
                    alignment_start_time = None
                    
                    reason = "time exceeded" if time_exceeded else "frames exceeded"
                    print(f"Alignment ended at {current_time:.2f}s ({reason})")
    
    # Handle any remaining alignment sequence at end of video
    if len(alignment_sequence) >= 2:
        sequence_data = process_alignment_sequence(alignment_sequence, fps)
        processed_sequences.append(alignment_sequence)
        print("Final alignment sequence processed")
    
    cap.release()
    cv2.destroyAllWindows()  # This closes all OpenCV windows
    
    print(f"Video processing complete. Found {len(processed_sequences)} sequences.")

    
    return processed_sequences
