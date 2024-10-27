import cv2
import face_recognition
import numpy as np
from pytubefix import YouTube

import os
import io
import tempfile
import logging
from typing import List, Any
from collections import defaultdict


from face_detection_utils import format_timestamps, save_face_image, ensure_dist_folder, write_to_file, save_results
from indexify.functions_sdk.data_objects import File
from indexify.functions_sdk.graph import Graph
from indexify.functions_sdk.image import Image
from indexify.functions_sdk.indexify_functions import indexify_function

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

yt_downloader_image = (
    Image().name("tensorlake/yt-downloader").run("pip install pytubefix")
)

face_recognition_image = (
    Image()
    .name("tensorlake/face_recognition")
    .run("pip install opencv-python cmake dlib setuptools face_recognition_models face_recognition numpy")
)

@indexify_function(image = yt_downloader_image)
def download_youtube_video(url: str) -> List[File]:
    """Download the YouTube video from the URL."""
    yt = YouTube(url)
    logging.info("Downloading video...")
    stream = yt.streams.filter(file_extension='mp4').get_highest_resolution() # download the highest res mp4
    
    buffer = io.BytesIO()
    stream.stream_to_buffer(buffer)
    video_bytes = buffer.getvalue()
    logging.info("Video downloaded")
    
    return [File(
        data=video_bytes,
        mime_type="video/mp4",
        metadata={
            'title': yt.title,
            'author': yt.author,
            'length': yt.length
        }
    )]

@indexify_function(image=face_recognition_image)
def process_video_for_faces(file: File) -> Any:
    """
    Process video file to detect and track unique faces with their timestamps.
    
    Args:
        file (File): File class instance containing video data
        sample_rate (int): Process every nth frame (default=1)
    
    Returns:
        dict: Dictionary containing face appearances with timestamps
    """
    # Ensure dist folder exists
    ensure_dist_folder()
    
    logging.info("Saving video to a temp file and opening it in open cv")
    
    # Create a temp file and save the video to it
    temp_file_path = tempfile.mktemp(suffix='.mp4')
    write_to_file(temp_file_path, file.data)

    # Decode the video from memory
    video = cv2.VideoCapture(temp_file_path)
    
    if not video.isOpened():
        raise ValueError("Failed to open video file")

    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Dictionary to store face encodings and their timestamps
    face_timestamps = defaultdict(list)
    face_locations_dict = {}  # Store face locations for saving images
    known_face_encodings = []
    known_face_indices = []
    best_face_frames = {}  # Store best quality face frame for each face
    face_qualities = defaultdict(float)  # Store face quality scores
    
    frame_number = 0
    sample_rate = 1 # TODO:- Make this configurable
    
    while video.isOpened():
        ret, frame = video.read()
        if not ret:
            break
            
        # Process every nth frame based on sample_rate
        if frame_number % sample_rate == 0:
            # Convert BGR to RGB for face_recognition
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Find faces in the frame
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
            
            # Current timestamp in seconds
            timestamp = frame_number / fps
            
            for face_encoding, face_location in zip(face_encodings, face_locations):
                # Check if we've seen this face before
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
                
                # Calculate face quality (using size as a simple metric)
                top, right, bottom, left = face_location
                face_size = (right - left) * (bottom - top)
                
                if True in matches:
                    # Face has been seen before
                    face_index = known_face_indices[matches.index(True)]
                    face_timestamps[face_index].append(timestamp)
                    
                    # Update best face frame if current is better quality
                    if face_size > face_qualities[face_index]:
                        face_qualities[face_index] = face_size
                        best_face_frames[face_index] = (frame.copy(), face_location)
                else:
                    # New face found
                    face_index = len(known_face_encodings)
                    known_face_encodings.append(face_encoding)
                    known_face_indices.append(face_index)
                    face_timestamps[face_index].append(timestamp)
                    face_qualities[face_index] = face_size
                    best_face_frames[face_index] = (frame.copy(), face_location)
        
        frame_number += 1
        
        # Display progress
        if frame_number % 100 == 0:
            progress = (frame_number / frame_count) * 100
            logging.info(f"Processing: {progress:.1f}% complete")
    
    video.release()
    
    # Save the best quality face image for each detected face
    for face_id, (frame, face_location) in best_face_frames.items():
        save_face_image(frame, face_location, face_id)
    
    # Convert timestamps to formatted strings and merge consecutive appearances
    formatted_results = format_timestamps(face_timestamps, fps)
    
    # Add face image paths to the results
    for face_id in formatted_results:
        formatted_results[face_id]['face_image'] = f'faces/face_{face_id}.jpg'
    
    logging.info(f"Finished processing the video {formatted_results}")
    save_results(formatted_results)
    return formatted_results
            
# TODO :- implement other modes of running this workflow.        
if __name__ == "__main__":
    # Create graph 
    logging.info("Starting workflow....")
    graph = Graph("Video_Face_Detection", start_node=download_youtube_video)
    graph.add_edge(download_youtube_video, process_video_for_faces)
    youtube_url = "https://www.youtube.com/shorts/13BnRQXZ7Kc"
    invocation_id = graph.run(block_until_done=True, url=youtube_url)
    logging.info(f"Invocation Id: {invocation_id}")
    logging.info("Workflow finished.")
    
    