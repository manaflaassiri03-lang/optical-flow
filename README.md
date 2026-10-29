# Optical Flow with OpenCV

Comparison of two optical flow methods applied to a traffic video.

## Methods

- **fb.py**: dense optical flow (Farneback). Computes motion for every pixel.
- **lk.py**: sparse optical flow (Lucas-Kanade). Tracks selected feature points.

## Installation

pip install -r requirements.txt

## Run

python fb.py
python lk.py

## Notes

Based on the OpenCV optical flow tutorial. Part of my work on video analysis
for motion and event detection.
