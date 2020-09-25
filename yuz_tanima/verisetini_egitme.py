import cv2
import numpy as np
from PIL import Image
import os

veri_yolu = 'veriseti'

tanima = cv2.face.LBPHFaceRecognizer_create()
bulma = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def YuzveEtiketAl(veri_yolu):

    yuz_yollari = [os.path.join(veri_yolu, f) for f in os.listdir(veri_yolu)]
    yuzornekleri = []
    IDler = []

    for yuz_yolu in yuz_yollari:

        PIL_resim = Image.open(yuz_yolu).convert('L')
        resim_np = np.array(PIL_resim, 'uint8')

        ID = int(os.path.split(yuz_yolu)[-1].split(".")[1])
        yuzler = bulma.detectMultiScale(resim_np)

        for (x, y, w, h) in yuzler:
            yuzornekleri.append(resim_np[y:y+h, x:x+w])
            IDler.append(ID)

    return yuzornekleri, IDler

print("\n YUZ VERILERI EGITILIYOR...")
yuzler,IDler = YuzveEtiketAl(veri_yolu)
tanima.train(yuzler, np.array(IDler))

tanima.write('egitim/egitimverisi.yml')

print("\n {0} TANE YUZ EGITILDI.".format(len(np.unique(IDler))))