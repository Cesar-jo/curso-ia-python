import cv2 as cv
import os
import imutils
modelo='Cesar'
ruta1= r'C:/Users/ternu/OneDrive/Escritorio/curso ia python/Data'
rutacompleta = ruta1 + '/'+ modelo
if not os.path.exists(rutacompleta):
    os.makedirs(rutacompleta)



camara=cv.VideoCapture(0)
ruidos=cv.CascadeClassifier(r'C:\Users\ternu\OneDrive\Escritorio\Redesneuronalespython\entrenamientos opencv\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml')
id=0
while True:
    respuesta,captura=camara.read()
    if respuesta==False:break
    captura=imutils.resize(captura,width=640)

    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=captura.copy()

    cara=ruidos.detectMultiScale(grises,1.3,5)

    for(x,y,e1,e2) in cara:
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (0,255,0),2)
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        rostrocapturado=cv.resize(rostrocapturado, (160,160),interpolation=cv.INTER_CUBIC)
        cv.imwrite(rutacompleta+'/imagen_{}.jpg'.format(id), rostrocapturado)
        id=id+1
    
    cv.imshow("Resultado rostro", captura)

    if id==350:
        break
camara.release()
cv.destroyAllWindows()