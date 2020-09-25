import cv2
import os

kamera = cv2.VideoCapture(0)

kamera.set(3, 640) # video genisligi
kamera.set(4, 480) # video yuksekligi

yuz_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

yuz_ID = input("\n BU KISI ICIN BIR ID BELIRLEYINIZ : ")

print("\n YUZ YAKALAMA BASLIYOR....")

say = 0

while True:

    ret, resim = kamera.read()
    gri = cv2.cvtColor(resim, cv2.COLOR_BGR2GRAY)
    yuzler = yuz_cascade.detectMultiScale(gri, 1.3, 5)

    for (x, y, w, h) in yuzler:

        cv2.rectangle(resim, (x,y), (x+w, y+h), (255, 0, 0), 2)
        say += 1
        cv2.imwrite("veriseti/kullanici." + str(yuz_ID) + '.' + str(say) + '.jpg',gri[y:y+h, x:x+w])
        cv2.imshow("KAYIT", resim)

    k = cv2.waitKey(100) & 0xff

    if k == 27:
        break
    elif say >= 10:
        break

print("\n KAYIT TAMAMLANDI....")
kamera.release()
cv2.destroyAllWindows()