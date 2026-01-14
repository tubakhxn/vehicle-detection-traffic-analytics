
# Vehicle Detection and Traffic Analytics System

**Creator/Developer:** tubakhxn

## Project Overview
This project is a complete, real-time vehicle detection and traffic analytics system built with Python. It uses computer vision to detect, track, and analyze vehicles in traffic videos or webcam feeds. The system provides:
- Vehicle detection (car, bus, truck, bike)
- Vehicle tracking with unique IDs
- Speed estimation and over-speeding flagging
- Real-time display of analytics
- Logging and visualization of results

## Tech Stack
- Python 3.9+
- OpenCV
- NumPy
- YOLOv8 (Ultralytics)
- SORT (Simple Online and Realtime Tracking)
- Matplotlib
- Pandas

## Features
- Load traffic video or webcam feed
- Detect vehicles (car, bus, truck, bike) using YOLOv8
- Track vehicles with SORT (unique IDs)
- Count vehicles (no double-counting)
- Estimate speed (distance/frame, FPS, pixel-to-meter calibration)
- Flag over-speeding vehicles
- Display bounding boxes, IDs, class, speed
- Log all data to CSV
- Generate bar graph of vehicle counts
- Robust to multiple vehicles and occlusion

## Speed Calculation
Speed is estimated by tracking the distance a vehicle's centroid travels between frames, normalized by FPS and a pixel-to-meter calibration factor.
**Formula:**
```
Speed (km/h) = (distance_pixels * pixel_to_meter * FPS_normalization) * 3.6
```

## How to Run
1. Ensure Python 3.9+ is installed.
2. Run:
	```
	python main.py --source <video_path_or_0_for_webcam>
	```
	Example for webcam:
	```
	python main.py --source 0
	```
3. Outputs:
	- Logs: `output/logs.csv`
	- Analytics graph: `output/graphs.png`

## How to Fork
1. On GitHub, click the **Fork** button at the top right of the repository page.
2. Clone your forked repository:
	```
	git clone https://github.com/<your-username>/<repo-name>.git
	```
3. Make your changes and push to your fork.

## Limitations & Assumptions
- Pixel-to-meter ratio is assumed (adjust in code for real-world calibration)
- Speed estimation is approximate
- Occlusion handling is basic
- Requires a CUDA-capable GPU for best performance
