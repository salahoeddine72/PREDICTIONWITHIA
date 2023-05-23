import argparse
import os
import cv2
import numpy as np
import torch
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from torch import nn
from torchvision import transforms, models

PATH = 'PATH'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

IMAGE_SIZE = 200


def preprocess_image(im):
    transform = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    im = cv2.resize(im, (IMAGE_SIZE, IMAGE_SIZE))
    im = torch.tensor(im).permute(2, 0, 1)
    im = transform(im / 255.)
    return im[None]


def get_model():
    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    model.avgpool = nn.Sequential(
        nn.Conv2d(512, 512, kernel_size=3),
        nn.MaxPool2d(2),
        nn.ReLU(),
        nn.Flatten()
    )

    class ageGenderClassifier(nn.Module):
        def __init__(self):
            super(ageGenderClassifier, self).__init__()
            self.intermediate = nn.Sequential(
                nn.Linear(2048, 512),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(512, 128),
                nn.ReLU(),
                nn.Dropout(0.4),
                nn.Linear(128, 64),
                nn.ReLU(),
            )
            self.age_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            self.gender_classifier = nn.Sequential(
                nn.Linear(64, 1),
                nn.Sigmoid()
            )

        def forward(self, x):
            x = self.intermediate(x)
            age = self.age_classifier(x)
            gender = self.gender_classifier(x)
            return gender, age

    model.classifier = ageGenderClassifier()
    return model


modeles = get_model()
modeles.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
modeles.eval()

# command line argument
ap = argparse.ArgumentParser()
ap.add_argument("--mode", help="display")
mode = ap.parse_args().mode

# Créer le module
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

#  est une commande qui désactive l'utilisation d'OpenCL dans la bibliothèque OpenCV
cv2.ocl.setUseOpenCL(False)

# dictionnaire pour attribuer a chaque label une emotion
emotion_dict = {0: "En colere", 1: "Degoute", 2: "Craintif", 3: "Heureux", 4: "Neutre", 5: "Triste", 6: "Surpris"}

# Ouvrir la camera
cap = cv2.VideoCapture(0)
while True:
    # Trouver la cascade pour dessiner une boite englobante autour du visage
    ret, frame = cap.read()
    if not ret:
        break
    facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = facecasc.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y - 50), (x + w, y + h + 10), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]

        face = frame[y:y + h, x:x + w]
        face = preprocess_image(face)
        gen, age = modeles(face)



        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
        prediction = model.predict(cropped_img)
        maxindex = int(np.argmax(prediction))
        cv2.putText(frame, emotion_dict[maxindex], (x, y-100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    cv2.LINE_AA)
        gen = "Femme" if gen.item() > 0.5 else "Homme"
        age = str(int(age.item() * 80))
        cv2.putText(frame, age + " ans", (x + 130, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)
        cv2.putText(frame, gen, (x , y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                    lineType=cv2.LINE_AA)

    cv2.imshow('Video', cv2.resize(frame, (1600, 960), interpolation=cv2.INTER_CUBIC))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
