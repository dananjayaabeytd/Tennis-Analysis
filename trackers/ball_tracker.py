from ultralytics import YOLO
import cv2
import pickle
import pandas as pd

class BallTracker:
    def __init__(self,model_path):
        self.model = YOLO(model_path)
        
    def interpolate_ball_positions(self, ball_positions):
        
        # Extract the ball positions associated with the key '1' from each entry in the list.
        # If the key '1' is not present, use an empty list as default.
        ball_positions = [x.get(1,[]) for x in ball_positions]
        
        # Convert the extracted ball positions into a Pandas DataFrame for easier processing.
        # The columns represent the coordinates of two points (x1, y1) and (x2, y2).
        df_ball_positions = pd.DataFrame(ball_positions,columns=['x1','y1','x2','y2'])

        # Fill in the missing values by interpolating between available values.
        df_ball_positions = df_ball_positions.interpolate()
        
        # Backfill any remaining missing values using the closest previous valid values.
        df_ball_positions = df_ball_positions.bfill()

        # Convert the DataFrame back into the original list format.
        # Each row is wrapped in a dictionary with the key '1'.
        ball_positions = [{1:x} for x in df_ball_positions.to_numpy().tolist()]

        # Return the cleaned and completed list of ball positions.
        return ball_positions
        
    def detect_frames(self,frames, read_from_stub=False, stub_path=None):
        ball_detections = []

        if read_from_stub and stub_path is not None:
            with open(stub_path, 'rb') as f:
                ball_detections = pickle.load(f)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)
        
        if stub_path is not None:
            with open(stub_path, 'wb') as f:
                pickle.dump(ball_detections, f)
        
        return ball_detections
        
    def detect_frame(self,frame):
        results = self.model.predict(frame,conf=0.15)[0]

        ball_dict = {}
        
        for box in results.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result
                
        return ball_dict
    
    def draw_bboxes(self,video_frames,player_detections):
        output_video_frames = []
        
        for frame,ball_dict in zip(video_frames,player_detections):
            
            #Draw bouding boxes
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                cv2.putText(frame, f"Ball ID: {track_id}",(int(bbox[0]),int(bbox[1] -10 )),cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 255), 2)
            output_video_frames.append(frame)
            
        return output_video_frames
                
    
    