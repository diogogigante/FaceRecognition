#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 24 17:38:20 2020

@author: diogo
"""


import face_recognition
import cv2
import numpy as np
import os

# This is a demo of running face recognition on live video from your webcam. It's a little more complicated than the
# other example, but it includes some basic performance tweaks to make things run a lot faster:
#   1. Process each video frame at 1/4 resolution (though still display it at full resolution)
#   2. Only detect faces in every other frame of video.

# PLEASE NOTE: This example requires OpenCV (the `cv2` library) to be installed only to read from your webcam.
# OpenCV is *not* required to use the face_recognition library. It's only required if you want to run this
# specific demo. If you have trouble installing it, try any of the other demos that don't require it instead.

menu = {}
menu['1']="Adicionar Pessoa" 
menu['2']="Executar"
menu['0']="Sair"

while True: 
    options=menu.keys()
    for entry in options: 
      print(entry, '->', menu[entry])

    selection=input("Opção:") 
    if selection =='1': 
        while True: 
            name=input("Nome (0 -> Cancelar):") 
            if name == '0':
                break
            elif len(name) < 4:
                print('Nome têm que conter no mínimo três letras')  
            else: 
                cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
                 # return a single frame in variable `frame`

                while(True):
                    ret,frame = cap.read()
                    img_name = "users/{}.png".format(name)

                    font                   = cv2.FONT_HERSHEY_SIMPLEX
                    bottomLeftCornerOfText = (10,500)
                    fontScale              = 1
                    fontColor              = (255,255,255)
                    lineType               = 2

                    cv2.putText(frame,'f -> Fotografar', 
                        bottomLeftCornerOfText, 
                        font, 
                        fontScale,
                        fontColor,
                        lineType)

                    cv2.imshow('Adicionar Pessoa',frame) #display the captured image
                    if cv2.waitKey(1) & 0xFF == ord('f'): #save on pressing 'y' 
                        cv2.imwrite(img_name,frame)
                        print('Pessoa',name,'adicionada com sucesso!')
                        break

                cap.release()
                cv2.destroyAllWindows()

                for i in range(10):
                    cv2.waitKey(1)

                print('')
                break
    elif selection == '2': 
        # Get a reference to webcam #0 (the default one)
        video_capture = cv2.VideoCapture(0)

        #from face_recognition.face_recognition_cli import image_files_in_folder
        my_dir = 'users/' # Folder where all your image files reside. Ensure it ends with '/
        known_face_encodings = [] # Create an empty list for saving encoded files
        known_face_names = []
        for i in os.listdir(my_dir): # Loop over the folder to list individual files
            image = my_dir + i
            nome = image.split('/')[1]
            nome = nome.split('.')[0]
            image = face_recognition.load_image_file(image) # Run your load command
            image_encoding = face_recognition.face_encodings(image) # Run your encoding command
            known_face_encodings.append(image_encoding[0]) # Append the results to encoding_for_file list
            known_face_names.append(nome)

        # Initialize some variables
        face_locations = []
        face_encodings = []
        face_names = []
        process_this_frame = True

        while True:
            # Grab a single frame of video
            ret, frame = video_capture.read()

            # Resize frame of video to 1/4 size for faster face recognition processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
            rgb_small_frame = small_frame[:, :, ::-1]

            # Only process every other frame of video to save time
            if process_this_frame:
                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

            process_this_frame = not process_this_frame


            # Display the results
            for (top, right, bottom, left), name in zip(face_locations, face_names):
                # Scale back up face locations since the frame we detected in was scaled to 1/4 size
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4

                # Draw a box around the face
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                # Draw a label with a name below the face
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                font = cv2.FONT_HERSHEY_DUPLEX
                cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

            font                   = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10,500)
            fontScale              = 1
            fontColor              = (255,255,255)
            lineType               = 2

            cv2.putText(frame,'0 -> Sair', 
                bottomLeftCornerOfText, 
                font, 
                fontScale,
                fontColor,
                lineType)

            # Display the resulting image
            cv2.imshow('Reconhecimento', frame)

            # Hit 'q' on the keyboard to quit!
            if cv2.waitKey(1) & 0xFF == ord('0'):
                break

        # Release handle to the webcam
        video_capture.release()
        cv2.destroyAllWindows()

        for i in range(10):
            cv2.waitKey(1)

        print('')
        break
        
    elif selection == '0': 
      break
    else: 
      print("Opção Inválida!") 
