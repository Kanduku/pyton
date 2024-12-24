import os
import time
from flask import Flask, render_template, request, send_from_directory
from pytube import YouTube
from moviepy.editor import VideoFileClip
import librosa
import numpy as np
import cv2
from datetime import datetime

app = Flask(__name__)

# Directory to store videos
VIDEO_FOLDER = "generated_videos"
STATIC_FOLDER = "static"
os.makedirs(VIDEO_FOLDER, exist_ok=True)

# Function to download and extract audio from YouTube video
def download_and_extract_audio(url, audio_filename="audio.mp3", video_filename="video.mp4"):
    yt = YouTube(url)
    video_stream = yt.streams.filter(only_audio=True).first()
    video_stream.download(filename=video_filename)
    
    # Extract audio from video using moviepy
    video_clip = VideoFileClip(video_filename)
    audio_clip = video_clip.audio
    audio_clip.write_audiofile(audio_filename)
    audio_clip.close()
    video_clip.close()

# Function to extract basic audio features (fast)
def extract_audio_features(audio_filename="audio.mp3"):
    y, sr = librosa.load(audio_filename, sr=22050)  # Lower sample rate for faster processing
    energy = librosa.feature.rms(y=y)
    return energy

# Function to generate video frame using basic motion based on energy
def generate_video_frame(i, energy, frame_size=(160, 120)):
    frame = np.zeros((frame_size[1], frame_size[0], 3), dtype=np.uint8)
    intensity = np.uint8(255 * energy[0, i] / np.max(energy))  # Normalize energy
    color = np.array([intensity, 255 - intensity, intensity], dtype=np.uint8)
    frame[:] = color  # Color entire frame with intensity
    return frame

# Function to generate motion video frames
def generate_motion_video_parallel(energy, video_filename="motion_video.mp4", frame_size=(160, 120), frame_rate=30):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, frame_rate, frame_size)
    
    for i in range(0, len(energy[0]), 10):  # Skip frames for speed (every 10th frame)
        frame = generate_video_frame(i, energy, frame_size)
        out.write(frame)
    
    out.release()

# Function to convert motion video to real video
def convert_to_real_video(motion_video_filename="motion_video.mp4", output_filename="real_video.mp4"):
    video_clip = VideoFileClip(motion_video_filename)
    video_clip.write_videofile(output_filename, codec="libx264", threads=4)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process_video', methods=['POST'])
def process_video():
    url = request.form['url']
    video_filename = os.path.join(VIDEO_FOLDER, "motion_video.mp4")
    output_filename = os.path.join(STATIC_FOLDER, "real_video.mp4")
    
    # Start the timer
    start_time = time.time()

    # Step 1: Download and extract audio from YouTube video
    download_and_extract_audio(url, audio_filename="audio.mp3", video_filename="video.mp4")
    
    # Step 2: Extract basic audio features (volume envelope)
    energy = extract_audio_features("audio.mp3")
    
    # Step 3: Generate motion video based on audio features
    generate_motion_video_parallel(energy, video_filename=video_filename)
    
    # Step 4: Convert motion video to real video format
    convert_to_real_video(video_filename, output_filename=output_filename)

    # Calculate the total time taken
    end_time = time.time()
    processing_time = end_time - start_time

    # Display the time for video generation
    return render_template('index.html', video_url=f"/static/{os.path.basename(output_filename)}", processing_time=processing_time)

@app.route('/static/<filename>')
def serve_video(filename):
    return send_from_directory(STATIC_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
