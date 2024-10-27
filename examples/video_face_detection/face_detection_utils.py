import cv2
import numpy as np

import json
import os
from datetime import timedelta
import shutil
from typing import Tuple

def format_timestamps(face_timestamps, fps, merge_threshold=1.0):
    """
    Format timestamps and merge consecutive appearances within threshold.
    """
    formatted_results = {}
    
    for face_id, timestamps in face_timestamps.items():
        appearances = []
        timestamps.sort()
        
        if not timestamps:
            continue
            
        start_time = timestamps[0]
        prev_time = timestamps[0]
        
        for time in timestamps[1:]:
            if time - prev_time > merge_threshold:
                # Convert to readable format
                start_str = str(timedelta(seconds=int(start_time)))
                end_str = str(timedelta(seconds=int(prev_time)))
                appearances.append({
                    "start": start_str,
                    "end": end_str,
                    "duration": f"{prev_time - start_time:.2f}s"
                })
                start_time = time
            prev_time = time
        
        # Add the last segment
        start_str = str(timedelta(seconds=int(start_time)))
        end_str = str(timedelta(seconds=int(prev_time)))
        appearances.append({
            "start": start_str,
            "end": end_str,
            "duration": f"{prev_time - start_time:.2f}s"
        })
        
        formatted_results[f"face_{face_id}"] = {
            "appearances": appearances,
            "total_appearances": len(appearances)
        }
    
    return formatted_results

def save_results(results):
    """
    Save the results to a JSON file in the dist folder.
    """
    output_path = 'dist/face_timestamps.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def write_to_file(file_path: str, data: bytes):
    '''
    Write some data to a file
    '''
    with open(file_path, 'wb') as temp_file:
        temp_file.write(data)
        temp_file.flush()
        os.fsync(temp_file.fileno())

def ensure_dist_folder():
    """Create dist folder if it doesn't exist and clean it if it does."""
    if os.path.exists('dist'):
        shutil.rmtree('dist')
    os.makedirs('dist')
    os.makedirs('dist/faces')
    
def save_face_image(frame: np.ndarray, face_location: Tuple[int, int, int, int], face_id: int):
    """
    Extract and save a face image from the frame.
    
    Args:
        frame: The video frame containing the face
        face_location: Tuple of (top, right, bottom, left) coordinates
        face_id: Identifier for the face
    """
    top, right, bottom, left = face_location
    # Add padding around the face (20% of face size)
    height = bottom - top
    width = right - left
    padding_v = int(height * 0.2)
    padding_h = int(width * 0.2)
    
    # Ensure padded coordinates don't go outside frame bounds
    frame_height, frame_width = frame.shape[:2]
    top = max(0, top - padding_v)
    bottom = min(frame_height, bottom + padding_v)
    left = max(0, left - padding_h)
    right = min(frame_width, right + padding_h)
    
    face_image = frame[top:bottom, left:right]
    face_path = f'dist/faces/face_{face_id}.jpg'
    cv2.imwrite(face_path, face_image)