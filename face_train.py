import cv2 as cv
import os
import numpy as np

#Listando o nome das pessoas que o modelo será treinado para reconhecer
DIR = r"C:\Users\DELL\Desktop\Face-Reco-openCV\train"
pessoas = [x for x in os.listdir(DIR)]


def prepara_treino(path_haar:str=r"haar_face.xml"):
    """
    Função que prepara o dataset para treinar no reconhecimento de rostos

    Entrada:
        - path_haar: caminho para o arquivo xml treina do em rostos (str)

    Saída
        - atributos: atributos de cada pessoa em que o modelo foi treinado (list)
        - tags: identificador de cada pessoa em que o modelo foi treinado (list)

    """ 
    haar_cascade = cv.CascadeClassifier(path_haar) #modelo treinado em reconhecimento de rostos
    atributos = [] #para guardar os atributos de cada pesosa
    tags = [] #para identificar cada pessoa

    for pessoa in pessoas:
        path = os.path.join(DIR, pessoa) #caminho para o diretório de uma pessoa
        tag = pessoas.index(pessoa) #identificador de cada pessoa

        for img in os.listdir(path):
            img_path = os.path.join(path, img) #caminho para uma imagem específica
            img_mat = cv.imread(img_path)
            cinza = cv.cvtColor(img_mat, cv.COLOR_BGR2GRAY) #treino feito em cima das fotos em GrayScale
            
            #instanciando o modelo base
            rec_rostos = haar_cascade.detectMultiScale(cinza, scaleFactor=1.1, minNeighbors=4)

            for (x, y, w, h) in rec_rostos:
                rosto_ri = cinza[y:y+h, x:x+w] #separa a região de interesse (rosto)
                atributos.append(rosto_ri)
                tags.append(tag)

    return atributos, tags


atr, ids = prepara_treino("haar_face.xml")
atr = np.array(atr, dtype="object")
ids = np.array(ids)

faceRec = cv.face.LBPHFaceRecognizer_create()

# Treino do modelo para reconhecimento
faceRec.train(atr, ids)

# Guardando resultados
faceRec.save("rosto_treinados.yml")
np.save("atributos.npy", atr)
np.save("ids.npy", ids)
