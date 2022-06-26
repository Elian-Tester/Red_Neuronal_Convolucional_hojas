import os
from pickletools import optimize
from pyexpat import model
from PIL import Image
from matplotlib import image
import numpy as np
import tensorflow as tf
from tensorflow import keras 
import matplotlib.pyplot as plt
import seaborn as sns


CATEGORIAS = []
IMAGENES = []
LABELS = []
HISTORIAL = []

def red_neuronal():
    print("\nRed Neuronal...")
    cargando_Imagenes()    
    
    dibujar()

    model = modelo()
    
    modeloEntrenado = entrenamiento(model[0], model[1], model[2])

    pruebas_muestra(modeloEntrenado)

    pred = cargando_img_test_modelo()
    
    matrizConfusion(model[0], pred[0], pred[1])
    
    graficar_Cuerva_Evolucion()

def dibujar():
    dibujar = plt.figure(figsize=(7, 7))

    for i in range( len(IMAGENES) ):
        plt.subplot(10,12,i+1)
        plt.xticks([])
        plt.yticks([])        
        plt.grid(False)
        plt.imshow(IMAGENES[i])        
    plt.show()


def cargando_Imagenes():
    print("\nCargando imagenes:")
    global CATEGORIAS
    CATEGORIAS = os.listdir("dataSet/entrenamiento")    
    print(CATEGORIAS)

    print("\nImagenes:")
    contaLabel = 0
    for tipo in CATEGORIAS:
        for img in os.listdir(f"dataSet/entrenamiento/{tipo}"):        
            imagen = Image.open(f"dataSet/entrenamiento/{tipo}/{img}").resize((100,100))
            imagen = imagen.convert("RGB")
            imagen = np.asarray(imagen)
            print(len(imagen.shape))
            #if(len(imagen.shape)==3):
                #imagen = imagen[:,:,0]
            print(imagen.shape)
            
            global IMAGENES
            global LABELS
            IMAGENES.append(imagen)
            LABELS.append(contaLabel)
        contaLabel+=1


def modelo():
    print("\nModelo: ")
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32,(3,3), input_shape=(100,100, 3), activation='relu'),
        #tf.keras.layers.Flatten(input_shape=(100,100)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64,(3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Flatten(),
        #tf.keras.layers.Dense(10, activation='relu'),
        tf.keras.layers.Dense(units=100, activation='relu'),
        tf.keras.layers.Dense(10, activation='softmax'),
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    imagenes_array = np.asarray(IMAGENES, dtype="float32")
    labels_array = np.asarray(LABELS, dtype="float32")
    
    return [model, imagenes_array, labels_array]
    #entrenamiento(model, imagenes_array, labels_array)

def entrenamiento(model, imagenes_array, labels_array):
    print("\nEntrenando: ")
    global HISTORIAL
    HISTORIAL = model.fit(imagenes_array, labels_array, epochs=40)
    
    return model
    #pruebas_muestra(model)

def pruebas_muestra(model):
    print("\nPrueba muestra:")
    print(CATEGORIAS)
    categoriaMuestra = os.listdir('dataSet/muestras')
    print(f"\nCategorias muestra: {categoriaMuestra}")
    
    for tipo in categoriaMuestra:
        for img in os.listdir(f'dataSet/muestras/{tipo}'):
            print(img)
            imagen = Image.open(f'dataSet/muestras/{tipo}/{img}').resize((100,100))            
            
            imagen = imagen.convert('RGB')
            imagen = np.asanyarray(imagen)
            #print(imagen)
            #if(len(imagen.shape)==3):
                #imagen = imagen[:,:,0]

            imagen = np.array([imagen])
            print("\nerror rango:___")
            print(imagen)
            prediccion = model.predict(imagen)
            print(prediccion)
            print(np.argmax(prediccion))
            print(CATEGORIAS)
            print(CATEGORIAS[np.argmax(prediccion)])

def cargando_img_test_modelo():
    print("\nTest imagenes:")
    
    categoriaTest = os.listdir('dataSet/test')
    print(f"\nCategorias test: {categoriaTest}")

    imagenes_test = []
    labels_test = []
    conta_test = 0

    for tipo in categoriaTest:
        for imagen in os.listdir(f'dataSet/test/{tipo}'):
            img = Image.open(f'dataSet/test/{tipo}/{imagen}').resize((100,100))
            img = img.convert('RGB')
            img = np.asanyarray(img)
            
            #if(len(img.shape)==3):
                #img = img[:,:,0]

            imagenes_test.append(img)
            labels_test.append(conta_test)
        conta_test+=1
    
    test_imagen = np.array(imagenes_test)
    test_label = np.array(labels_test)

    return [test_imagen, test_label]


def matrizConfusion(model, test_imagen, test_label):
    print("\nMatriz confusion: ")
    names = ['Crescentia','Epazote', 'Ixora', 'Limon', 'Platano']
    y_prediccion = np.argmax(model.predict(test_imagen), axis=1)
    y_true = np.array(test_label)

    print("\nPrediccion: ")
    print(f"\nPrediccion Y: {y_prediccion}")
    print(f"\nVerdadero Y: {y_true}")

    confusion_matriz = tf.math.confusion_matrix(y_true, y_prediccion)
    fig_matriz = plt.figure(figsize=(9,5))
    sns.heatmap(
        confusion_matriz,
        xticklabels= names, 
        yticklabels=names, 
        annot=True, 
        cmap='icefire', 
        fmt='g'
        )
    plt.xlabel('Prediccion')
    plt.ylabel('Verdadero')
    plt.show()


def graficar_Cuerva_Evolucion():
    print("Graficando curva de error")
    fig = plt.figure(figsize=(9,5))
    plt.plot(HISTORIAL.history['loss'])
    plt.title('Evolucion de error')
    plt.xlabel('Epocas')
    plt.ylabel('Valor error')
    plt.show()


if __name__=="__main__":
    print("Hola")
    red_neuronal()