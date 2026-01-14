from utils import get_centroid, PIXEL_TO_METER

class SpeedEstimator:
    def __init__(self, fps, pixel_to_meter=PIXEL_TO_METER):
        self.fps = fps
        self.pixel_to_meter = pixel_to_meter
        self.last_positions = {}  # id: (centroid, frame)

    def estimate(self, track_id, bbox, frame_num):
        centroid = get_centroid(bbox)
        if track_id in self.last_positions:
            prev_centroid, prev_frame = self.last_positions[track_id]
            dist_pixels = ((centroid[0] - prev_centroid[0]) ** 2 + (centroid[1] - prev_centroid[1]) ** 2) ** 0.5
            frames_elapsed = frame_num - prev_frame
            if frames_elapsed > 0:
                speed_mps = (dist_pixels * self.pixel_to_meter * self.fps) / frames_elapsed
                speed_kmh = speed_mps * 3.6
            else:
                speed_kmh = 0
        else:
            speed_kmh = 0
        self.last_positions[track_id] = (centroid, frame_num)
        return speed_kmh
