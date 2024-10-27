# Video Face Detection with Indexify

This project demonstrates how to detect and identify unique faces in a video using Indexify. The pipeline downloads a video,
reads it into open-cv and analyses the video frame by frame for unique saves and also saves the timestamp and duration the face was in the video.

## Prerequisites

- Python 3.9+

## Installation and Usage

### Option 1: Local Installation - In Process

1. Clone this repository:
   ```
   git clone https://github.com/tensorlakeai/indexify
   cd indexify/examples/video_face_detection
   ```

2. Create a virtual environment and activate it:
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Run the main script:
   ```
   python workflow.py
   ```

## How it Works
### Setup and Initialization:
Creates a 'dist' folder structure for outputs (`faces/` subfolder for images and `face_timestamps.json` for the timestamps)
Takes input as either a YouTube URL (using pytubefix) or video file
Converts video input into a format OpenCV can process

### Main Processing Loop (process_video_for_faces function):
Reads video frame by frame
For each processed frame:
- Converts frame from BGR to RGB color space
- Detects faces using face_recognition library
- Extracts face encodings (numerical representations of faces)

### Face Recognition and Tracking:
For each detected face in a frame:

- Compares face encoding with previously seen faces
- If face matches existing face: adds timestamp to that face's records
- If new face: creates new face entry with initial timestamp
- Keeps track of best quality face image for each unique person
- Uses face size as quality metric
- Updates stored face image when better quality found
