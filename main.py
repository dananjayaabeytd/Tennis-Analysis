import os
from trackers import PlayerTracker
from utils import (
    read_video,
    save_video
    )


# Set environment variable to avoid OpenMP error
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    
   # Read Video
    input_video_path = "input_videos/input_video.mp4"
    video_frames = read_video(input_video_path)
    
    player_tracker = PlayerTracker(model_path="yolov8x")
    
    player_detections = player_tracker.detect_frames(video_frames,read_from_stub=False,stub_path="tracker_stubs/player_detections.pkl")
                  
                  
    # Draw Output
    # Draw Player bounding boxes
    output_video_frames = player_tracker.draw_bboxes(video_frames, player_detections)                                                
    
    save_video(video_frames, "output_videos/output_video.avi")
    
if __name__ == "__main__":
    main()