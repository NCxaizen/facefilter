import cv2,os,urllib.request
import numpy as np
from ant import facedetection as fd

class VideoCamera(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
    
    def __del__(self):
        self.video.release()
    
    def get_frame(self):
        left_eye = cv2.imread('media/dragon_right_eye.png')
        right_eye = cv2.imread('media/dragon_left_eye.png')

        # Initialize the VideoCapture object to read from the smoke animation video stored in the disk.
        smoke_animation = cv2.VideoCapture('media/smoke_animation.mp4')

        # Set the smoke animation video frame counter to zero.
        smoke_frame_counter = 0
            
        # Read a frame.
        ok, frame = self.video.read()
        
            
        # Read a frame from smoke animation video
        _, smoke_frame = smoke_animation.read()
        
        # Increment the smoke animation video frame counter.
        smoke_frame_counter += 1
        
        # Check if the current frame is the last frame of the smoke animation video.
        if smoke_frame_counter == smoke_animation.get(cv2.CAP_PROP_FRAME_COUNT):     
            
            # Set the current frame position to first frame to restart the video.
            smoke_animation.set(cv2.CAP_PROP_POS_FRAMES, 0)
            
            # Set the smoke animation video frame counter to zero.
            smoke_frame_counter = 0
        
        # Flip the frame horizontally for natural (selfie-view) visualization.
        frame = cv2.flip(frame, 1)
        
        # Perform Face landmarks detection.
        _, face_mesh_results = fd.detectFacialLandmarks(frame, fd.face_mesh_videos, display=False)
        
        # Check if facial landmarks are found.
        if face_mesh_results.multi_face_landmarks:
            
            # Get the mouth isOpen status of the person in the frame.
            _, mouth_status = fd.isOpen(frame, face_mesh_results, 'MOUTH', 
                                        threshold=15, display=False)
            
            # Get the left eye isOpen status of the person in the frame.
            _, left_eye_status = fd.isOpen(frame, face_mesh_results, 'LEFT EYE', 
                                            threshold=4.5 , display=False)
            
            # Get the right eye isOpen status of the person in the frame.
            _, right_eye_status = fd.isOpen(frame, face_mesh_results, 'RIGHT EYE', 
                                            threshold=4.5, display=False)
            
            # Iterate over the found faces.
            for face_num, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                
                # Check if the left eye of the face is open.
                if left_eye_status[face_num] == 'OPEN':
                    
                    # Overlay the left eye image on the frame at the appropriate location.
                    frame = fd.overlay(frame, left_eye, face_landmarks,
                                    'LEFT EYE', fd.mp_face_mesh.FACEMESH_LEFT_EYE, display=False)
                
                # Check if the right eye of the face is open.
                if right_eye_status[face_num] == 'OPEN':
                    
                    # Overlay the right eye image on the frame at the appropriate location.
                    frame = fd.overlay(frame, right_eye, face_landmarks,
                                    'RIGHT EYE', fd.mp_face_mesh.FACEMESH_RIGHT_EYE, display=False)
                
                # Check if the mouth of the face is open.
                if mouth_status[face_num] == 'OPEN':
                    
                    # Overlay the smoke animation on the frame at the appropriate location.
                    frame = fd.overlay(frame, smoke_frame, face_landmarks, 
                                    'MOUTH', fd.mp_face_mesh.FACEMESH_LIPS, display=False)
                
                #return frames as jpeg
                ret, jpeg = cv2.imencode('.jpg', frame)
                return jpeg.tobytes()
        