import face_recognition
import cv2
import os
from os import listdir
from os.path import isfile, join
import time
import numpy as np

cwd = os.getcwd()

onlyfiles = [f for f in listdir(cwd) if isfile(join(cwd, f)) and (('jpg' in f) or ('JPG' in f))]
number_of_files = len(onlyfiles)
print(number_of_files)

# This is a demo of running face recognition on a video file and saving the results to a new video file.
#
# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

# Open the input movie file
input_movie = cv2.VideoCapture("MVI_1045.MP4")
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# # Create an output movie file (make sure resolution/frame rate matches input video!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter('2111_try.avi', fourcc, 1, (1920, 1080))

known_faces = []
faces_list = ["aakash", "alex", "ama", "chinmay", "gabi", "keshav", "kevin", "kwu", "maryyann", "nick", "sagar", "yimin"]
index_to_faces = []

# capture file names and match it to the subjects' name list
for file in onlyfiles:
    for face in faces_list:
        if face in file.lower():
            index_to_faces.append(face)
        continue

print(index_to_faces)

print(onlyfiles)
# Load batches of images
i = 0
for i in range(number_of_files):
    known_face = face_recognition.load_image_file(onlyfiles[i])
    known_face_encoding = face_recognition.face_encodings(known_face)[0]
    known_faces.append(known_face_encoding)
    print(onlyfiles[i] + " encoded successfully")

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []
frame_number = 0

name_cnt = np.zeros(len(faces_list))

start_time = time.time()
while True:
    # Grab a single frame of video
    input_movie.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
    ret, frame = input_movie.read()

    # Quit when the input video file ends
    if not ret:
        break

    print("\nStarting frame: " + str(frame_number))
    print("Execution time for reading frame--- %s seconds ---" % (time.time() - start_time))
    # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
    rgb_frame = frame[:, :, ::-1]

    # Find all the faces and face encodings in the current frame of video
    face_locations = face_recognition.face_locations(rgb_frame, number_of_times_to_upsample=0, model="cnn")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    cnt1 = 0

    for face_encoding in face_encodings:
        # print('here')
        # See if the face is a match for the known face(s)
        # print(known_faces.__sizeof__())
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        # If you had more than 2 faces, you could make this logic a lot prettier
        # but I kept it simple for the demo
        name = None
        for match_index in range(number_of_files):
            if match[match_index] and name is None:
                name = index_to_faces[match_index]
                name_cnt[faces_list.index(name)] += 1
        if name is None:
            name = "UNKNOWN"

        face_names.append(name)
        cnt1 = cnt1 + 1

    # Label the results
    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Draw a box around the face
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Draw a label with a name below the face
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1, (255, 255, 255), 1)

    # Write the resulting image to the output video file
    print("Writing frame {} / {}".format(frame_number, length))
    output_movie.write(frame)

    print("Execution time for processing frame--- %s seconds ---" % (time.time() - start_time))
    frame_number += 2

ii = 0
print("\n")
for faces1 in faces_list:
    print("The number of times " + faces1 + " appears is: " + str(name_cnt[ii]))
    ii = ii + 1

# All done!
input_movie.release()
cv2.destroyAllWindows()
