# coding: latin-1
from tkinter import *
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
import time
import mediapipe as mp
import matplotlib.image as image
from PIL import Image, ImageOps
from numpy import asarray
import threading
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense,Dropout
from tensorflow.keras.callbacks import TensorBoard
from keras.optimizers import SGD
from PIL import Image,ImageTk

def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)#Color conversion BGR 2 RGB
    image.flags.writeable = False #Image is unwriteable
    results = model.process(image)#Make prediction
    image.flags.writeable = True#Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)#color conversion RGB 2 BGR
    return image, results

def draw_landmarks(image, results):
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS)#Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)#Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)#Draw hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)#Draw hand connections

def getModel():
    model = Sequential()
    model.add(LSTM(64, return_sequences=True, activation='tanh', input_shape=(30,1662)))
    model.add(Dropout(0.2))
    model.add(LSTM(128, return_sequences=True, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(LSTM(64, return_sequences=False, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    #model.add(Dense(actions.shape[0], activation='softmax'))
    model.add(Dense(actions.shape[0], activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.load_weights('finalModel60fpsfinal12.h5')
    return model

def draw_styled_landmarks(image, results):
    #Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80, 110, 10), thickness = 1, circle_radius =1),
                             mp_drawing.DrawingSpec(color=(80, 256, 121), thickness = 1, circle_radius =1)
                             )
    #Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80, 22, 10), thickness = 2, circle_radius =4),
                             mp_drawing.DrawingSpec(color=(80, 44, 121), thickness = 2, circle_radius =2)
                             )
    #Draw hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121, 22, 76), thickness = 2, circle_radius =4),
                             mp_drawing.DrawingSpec(color=(121, 44, 250), thickness = 2, circle_radius =2)
                             )
    #Draw hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245, 117, 66), thickness = 2, circle_radius =4),
                             mp_drawing.DrawingSpec(color=(245, 66, 230), thickness = 2, circle_radius =2)
                             )

def extract_keypoints(results):
    face = np.array([[res.x, res.y, res.z] for res in results.face_landmarks.landmark]).flatten() if \
        results.face_landmarks else np.zeros(468*3)
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if \
        results.pose_landmarks else np.zeros(33*4)
    lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if \
        results.left_hand_landmarks else np.zeros(21*3)
    rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if \
        results.right_hand_landmarks else np.zeros(21*3)
    return np.concatenate([pose, face, lh, rh])


colors = [(0, 215, 255), (192, 192, 192), (50, 127, 205), (142, 241, 53), (123, 123, 133), (214, 122, 141),
          (214, 122, 141), (214, 122, 141),
          (214, 122, 141), (214, 122, 141), (214, 122, 141), (214, 122, 141), (214, 122, 141), (214, 122, 141),
          (214, 122, 141), (214, 122, 141),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (142, 241, 53), (123, 123, 133), (214, 122, 141),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (142, 241, 53), (123, 123, 133), (214, 122, 141),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (142, 241, 53), (123, 123, 133), (214, 122, 141),
          (245, 117, 16), (117, 245, 16), (16, 117, 245), (142, 241, 53), (123, 123, 133), (214, 122, 141)]


def prob_viz(res, actions, input_frame, colors):
    output_frame = input_frame.copy()
    count = 0
    res_arr = np.array(res)
    idx = np.argsort(res_arr)
    idx = idx[::-1]
    res_arr = res_arr[idx]
    new_actions = actions[idx]
    #     for num, prob in enumerate(res):
    #         cv2.rectangle(output_frame, (0,60+num*20), (int(prob*100), 90+num*20), colors[num], -1)
    #         cv2.putText(output_frame, actions[num].split('_')[0], (0, 85+num*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)
    #         cv2.putText(output_frame, str(int(prob*100)), (150, 85+num*20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 2, cv2.LINE_AA)

    cv2.putText(output_frame, 'Action', (0, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_frame, 'Percentage', (100, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)

    for num, prob in enumerate(sorted(res, reverse=True)):
        count += 1
        cv2.rectangle(output_frame, (0, 70 + num * 20), (int(prob * 100), 90 + num * 20), colors[num], -1)
        cv2.putText(output_frame, new_actions[num].split('_')[0], (0, 85 + num * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(output_frame, str("{:.2f}".format(prob * 100)) + ' %', (110, 85 + num * 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
        if count == 3:
            break

    return output_frame


def gg(model):
    # 1. New detection variables
    sequence = []
    sentence = ['']
    threshold = 0.75
    predictions = []
    no_of_frame = 0
    a = 0
    cap = cv2.VideoCapture(0)
    # Set mediapipe model
    start_time = time.time()
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Read feed
            ret, frame = cap.read()
            no_of_frame += 1
            # Make detections
            image, results = mediapipe_detection(frame, holistic)
            print(results)

            # Draw landmarks
            draw_styled_landmarks(image, results)

            # 2. Prediction logic
            keypoints = extract_keypoints(results)
    #         sequence.insert(0,keypoints)
    #         sequence = sequence[:30]
            sequence.append(keypoints)
            sequence = sequence[-30:]

            if len(sequence) == 30:
                res = model.predict(np.expand_dims(sequence, axis=0))[0]
                print(actions[np.argmax(res)])
                predictions.append(np.argmax(res))

            #3. Viz logic
                if np.unique(predictions[-15:])[0]==np.argmax(res):
                    if res[np.argmax(res)] > threshold:
                        #if len(sentence) > 0:
                            if actions[np.argmax(res)].split('_')[0] != sentence[-1]:
                                if actions[np.argmax(res)].split('_')[0] == 'Wait' and sentence[-1]!='':#actions[0].split('_')[0]
                                    sentence.append('') #actions[np.argmax(res)].split('_')[0]
                                #elif sentence[-1] == ' ':#actions[0].split('_')[0]
                                elif actions[np.argmax(res)].split('_')[0] != 'Wait' and sentence[-1]=='':
                                    sentence.append(actions[np.argmax(res)].split('_')[0])
    #                             elif np.unique(predictions[-90:])[0] == np.argmax(res) and a > 4 and actions[np.argmax(res)].split('_')[0] != 'wait':
    #                                 sentence.append(actions[np.argmax(res)].split('_')[0])

                       # else:
                           # sentence.append(actions[np.argmax(res)].split('_')[0])

                if len(sentence) > 7:
                    sentence = ['']

                # Viz probabilities
                image = prob_viz(res, actions, image, colors)

            #cv2.putText(image, "{}".format(no_of_frame), (600, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.rectangle(image, (0,0), (640, 40), (129, 196, 26), -1)
            cv2.putText(image, ' '.join(sentence), (3,30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
            if no_of_frame == 30:
                no_of_frame = 0
                a += 1
            # Show to screen
            image = cv2.resize(image,(960,720))
            cv2.imshow('OpenCV Feed', image)

            # Break gracefully

            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
    end_time = time.time()


def showMember():
    window2 = Toplevel()
    window2.title(" DANCHUBE'S HALL OF FAME ")
    window2.geometry("980x720")
    newlabel = Label(window2, text=" DANCHUBE'S HALL OF FAME ", font=('HERSHEY SIMPLEX', 30)).place(x=120, y=0)
    image = Image.open('TDST.png')
    image.thumbnail((250, 250), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label_image = Label(window2, image=photo).place(x=140, y=110)
    newlabel = Label(window2, text=" Ta Dinh Son Tung ", font=('HERSHEY SIMPLEX', 20)).place(x=120, y=70)

    image2 = Image.open('NHP.png')
    image2.thumbnail((250, 250), Image.ANTIALIAS)
    photo2 = ImageTk.PhotoImage(image2)
    label_image2 = Label(window2, image=photo2).place(x=620, y=110)
    newlabel = Label(window2, text=" Nguyen Ha Phan ", font=('HERSHEY SIMPLEX', 20)).place(x=610, y=70)

    image3 = Image.open('TTD.png')
    image3.thumbnail((250, 250), Image.ANTIALIAS)
    photo3 = ImageTk.PhotoImage(image3)
    label_image3 = Label(window2, image=photo3).place(x=130, y=450)
    newlabel = Label(window2, text=" Tran Trung Dinh ", font=('HERSHEY SIMPLEX', 20)).place(x=130, y=410)

    image4 = Image.open('CMS.png')
    image4.thumbnail((250, 250), Image.ANTIALIAS)
    photo4 = ImageTk.PhotoImage(image4)
    label_image4 = Label(window2, image=photo4).place(x=630, y=450)
    newlabel = Label(window2, text=" Cao Minh Son ", font=('HERSHEY SIMPLEX', 20)).place(x=630, y=410)
    window.mainloop()


def chooseProject():
    window2 = Toplevel()
    window2.title(" Choose Project ")
    window2.geometry("980x720")
    newlabel = Label(window2, text=" CHOOSE A PROJECT ", font=('HERSHEY SIMPLEX', 20)).place(x=350, y=10)
    image = Image.open('ASL.jpeg')
    image.thumbnail((350, 350), Image.ANTIALIAS)
    photo = ImageTk.PhotoImage(image)
    label_image = Label(window2, image=photo).place(x=120, y=70)
    button1 = Button(window2, text="Sign Language Recognition", width=25, command=SignLanguageRecognition).place(x=150,
                                                                                                                 y=340)

    image2 = Image.open('Compress.jfif')
    image2.thumbnail((350, 350), Image.ANTIALIAS)
    photo2 = ImageTk.PhotoImage(image2)
    label_image2 = Label(window2, image=photo2).place(x=600, y=70)
    button2 = Button(window2, text="Image compression using PCA", width=25).place(x=640, y=340)

    image3 = Image.open('Question.jfif')
    image3.thumbnail((350, 350), Image.ANTIALIAS)
    photo3 = ImageTk.PhotoImage(image3)
    label_image3 = Label(window2, image=photo3).place(x=125, y=400)
    button3 = Button(window2, text=" Coming soon ").place(x=185, y=670)

    image4 = Image.open('Question.jfif')
    image4.thumbnail((350, 350), Image.ANTIALIAS)
    photo4 = ImageTk.PhotoImage(image4)
    label_image4 = Label(window2, image=photo4).place(x=620, y=400)
    button4 = Button(window2, text=" Coming soon ").place(x=695, y=670)
    window.mainloop()

def haha():
    window2=Tk()
    window2.title(" Credit ")
    window2.geometry("750x720")
    newlabel = Label(window2,text = " CREDIT ",font=('HERSHEY SIMPLEX',20)).place(x=300,y=50)
    newlabel = Label(window2,text = " Idea ",font=('HERSHEY SIMPLEX',20)).place(x=50,y=100)
    newlabel = Label(window2,text = "Ta Dinh Son Tung ",font=('HERSHEY SIMPLEX',20)).place(x=470,y=100)
    newlabel = Label(window2,text = " Code ",font=('HERSHEY SIMPLEX',20)).place(x=50,y=200)
    newlabel = Label(window2,text = "Ta Dinh Son Tung ",font=('HERSHEY SIMPLEX',20)).place(x=470,y=200)
    newlabel = Label(window2,text = " Design ",font=('HERSHEY SIMPLEX',20)).place(x=50,y=300)
    newlabel = Label(window2,text = "Ta Dinh Son Tung ",font=('HERSHEY SIMPLEX',20)).place(x=470,y=300)
    newlabel = Label(window2,text = " Financial support ",font=('HERSHEY SIMPLEX',20)).place(x=50,y=400)
    newlabel = Label(window2,text = "Ta Dinh Son Tung ",font=('HERSHEY SIMPLEX',20)).place(x=470,y=400)
    newlabel = Label(window2,text = " Emotional support ",font=('HERSHEY SIMPLEX',20)).place(x=50,y=500)
    newlabel = Label(window2,text = "Ta Dinh Son Tung ",font=('HERSHEY SIMPLEX',20)).place(x=470,y=500)
    newlabel = Label(window2,text = " THANK YOU SO MUCH",font=('HERSHEY SIMPLEX',20)).place(x=210,y=600)
    newlabel = Label(window2,text = "Ta Dinh Son Tung",font=('HERSHEY SIMPLEX',20)).place(x=250,y=650)
    window.mainloop()

def SignLanguageRecognition():
    model=getModel()
    gg(model)

mp_holistic = mp.solutions.holistic #Holistic model
mp_drawing = mp.solutions.drawing_utils #Drawing utilities
actions = np.array(['Walk_', 'Wait_', 'Hello_L','Hello_R','Name_','No_','Sorry_R','Thanks_L',
                    'Fire_','Give up_','House_','Ready_','Drink_','Sleep_','Where_'])

window=Tk()
window.title(" Project Team DanChuBe ")
window.geometry("600x300")

image=Image.open("blue.png")
image=image.resize((600,300))
bg = ImageTk.PhotoImage(image)
label1 = Label(image=bg)
label1.place(x = 0, y = 0)


newlabel = Label(text = " DAN CHU BE ",font=('HERSHEY SIMPLEX',20),fg='red').place(x=205,y=50)
newlabel2 = Label(text = " AI1605 ",font=('HERSHEY SIMPLEX',15),fg='red').place(x=260,y=95)
button1 = Button(text = "Projects",font=10,bg='yellow',width=15,height=2,command=chooseProject,fg='blue',activebackground='yellow').place(x = 60,y = 150)
button2 = Button(text = "Members",font=10,width=15,height=2,command=showMember,fg='blue',bg='yellow').place(x = 360,y = 150)
button3 = Button(text = "Credit",width=10,command=haha,activebackground='black').place(x = 255,y = 250)
window.mainloop()
input()