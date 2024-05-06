import numpy as np
import cv2 as cv
import os

#init vars
haar_cascade = cv.CascadeClassifier("haar_face.xml")
#atr = np.load("atributos.npy", allow_pickle=True)
#ids = np.load("ids.npy")
pessoas = os.listdir(r"C:\Users\DELL\Desktop\Face-Reco-openCV\Faces\train")
faceRec = cv.face.LBPHFaceRecognizer_create()
faceRec.read("rostos_treinados.yml")

#inserir o caminho para uma imagem que será reconhecida 
path_to_img = r""
img = cv.imread(path_to_img)
cinza = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("Foto original", cinza)

#separa rostos na imagem
rosto = haar_cascade.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=4)

for (x, y, w, h) in rosto:
    rosto_ri = cinza[y:y+h, x:x+w] #separa região de interesse (rosto)

    tag, confianca = faceRec.predict(rosto_ri)
    print(f"Id = {pessoas[tag]} com confianca de {confianca}")

    cv.putText(img, str(pessoas[tag]), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0,200,0), thickness=2)
    cv.rectangle(img, (x, y), (x+w, y+h), (0,200,0), thickness=2)


cv.imshow("Rosto detectado", img)
cv.waitKey(0)