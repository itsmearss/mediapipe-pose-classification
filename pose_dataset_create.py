import cv2
import os
import csv
import numpy as np
import mediapipe as mp
import pandas as pd 

def create_pose_csv(cap, create_csv):
    ''' Create pose detections csv with a video.'''
    
    # Define colors for landmarks and connections
    color_pose1 = (245,117,66)
    color_pose2 = (245,66,230)

    if (cap.isOpened() == False):
        print("\nError opening the video file.")
        return
    else:
        pass

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True   
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                # Draw Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=color_pose1, thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=color_pose2, thickness=2, circle_radius=2)
                )

                try:
                    num_coords = len(results.pose_landmarks.landmark) # num_coords: 33

                    landmarks = ['class'] # Create first rows data.
                    for val in range(1, num_coords+1):
                        landmarks += ['x{}'.format(val), 'y{}'.format(val), 'z{}'.format(val), 'v{}'.format(val)]
                    
                    # Define first class rows in csv file.
                    with open(create_csv, mode='w', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(landmarks)
                except:
                    pass

                cv2.imshow('Raw Video Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break

    print(f'\nCreate {create_csv} done! \n\nNow you can run again.')
    cap.release()
    cv2.destroyAllWindows()

def add_record_coordinates(cap, class_name, export_csv):
    ''' Add recorded pose coordinates to an existing CSV file.'''
    
    if (cap.isOpened() == False):
        print("Error opening the video file.")
    else:
        input_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f'Frames per second: {input_fps}')
        print(f'Frame count: {frame_count}')

    mp_drawing = mp.solutions.drawing_utils # Drawing helpers.
    mp_holistic = mp.solutions.holistic     # Mediapipe Solutions.

    # Initiate holistic model
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        
        while cap.isOpened():
            ret, frame = cap.read()
            if ret == True:
                # Recolor Feed
                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False        

                # Make Detections
                results = holistic.process(image)

                # Recolor image back to BGR for rendering
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                # Draw Pose Detections
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                    mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                )

                # Export coordinates
                try:
                    # Extract Pose landmarks
                    pose = results.pose_landmarks.landmark
                    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())

                    # Concate rows
                    row = pose_row

                    # Append class name.
                    row.insert(0, class_name)

                    # Export to CSV
                    with open(export_csv, mode='a', newline='') as f:
                        csv_writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                        csv_writer.writerow(row) 

                except:
                    pass

                cv2.imshow('Raw Webcam Feed', image)

                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            else:
                break
            
    print('Add done!\n -------------------')
    cap.release()
    cv2.destroyAllWindows()
    check_csv_contents(file=export_csv)

def check_csv_contents(file):
    '''Print the top and bottom 5 rows of a CSV file.'''
    
    df = pd.read_csv(file)
    print(f'Top5 datas: \n{df.head()}')
    print(f'Last5 datas: \n{df.tail()}')

if __name__ == '__main__':
    
    # Add 3 categories of pose: cat_camel, bridge, heel_raise.
    add_class = 'Warrior-Pose'
    video_file_name = '1m'
    folder_name = 'Warrior-Pose'
    dataset_csv_file = './dataset/coords_dataset2.csv'

    video_path = "./videos/" + folder_name + "/" + video_file_name +".mp4"
    output_video = video_file_name + "_out.mp4"

    cap = cv2.VideoCapture(video_path)

    if os.path.isfile(dataset_csv_file):
        print (f'{dataset_csv_file}: Exist.')
        print(f'Add class: {add_class} \n-----------------')

        add_record_coordinates(cap=cap, class_name=add_class, export_csv=dataset_csv_file)
    else:
        print (f'{dataset_csv_file}: Not exist.')
        print('\nInitiate creating a csv file....\n')

        create_pose_csv(cap, create_csv=dataset_csv_file)
