import cv2
import numpy as np
import os

yoklama = open("YOKLAMA_LISTESI.txt", 'w+')
yoklama.write("                      YOKLAMA LISTESI   \n \n")

tanima = cv2.face.LBPHFaceRecognizer_create()
tanima.read('egitim/egitimverisi.yml')

yuz_cascade =cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

font = cv2.FONT_HERSHEY_SIMPLEX

ID = 5

isimler = ['','furkan','aydın','seyfi','levent','ibrahim']

kamera = cv2.VideoCapture(0)
kamera.set(3, 640)
kamera.set(4, 480)

minW = 0.1*kamera.get(3)
minH = 0.1*kamera.get(4)

while True:
    ret, resim = kamera.read()
    gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)

    yuzler = yuz_cascade.detectMultiScale(
        gri,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize= (int(minW), int(minH)),
    )

    for (x, y, w, h) in yuzler:

        cv2.rectangle(resim, (x,y), (x+w, y+h), (0,255,0), 2)

        ID, tahmin = tanima.predict(gri[y:y+h, x:x+w])

        if tahmin < 100:
            ID = isimler[ID]
            tahmin = " %{0}".format(round(100 - tahmin))
        else:
            ID = 'TANINMADI'
            tahmin = "%{0}".format(round(100 - tahmin))

        cv2.putText(resim, str(ID), (x+5, y-5), font, 1, (255,255,255), 2)
        cv2.putText(resim, str(tahmin), (x+5, y+h-5), font, 1, (255,255,0), 1)

        for i in range(10):
            yoklama.write(str(i+1) + '.KISI : ' + str(ID) + "\n")


    cv2.imshow("GORUNTU", resim)

    k = cv2.waitKey(10) & 0xff
    if k == 27:
        break

yoklama.close()
kamera.release()
cv2.destroyAllWindows()