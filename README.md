# Reconocimiento de dígitos
Está aplicación fue desarrollada para reconocer dígitos utilizando una cámara. El modelo computacional corresponde a una red neuronal convolucional entrenada con el conjunto de datos MNITS.
## Requerimientos
- Python 3.5 a 3.8
- Tensorflow
- Keras
- Opencv
```sh
$ pip3 install tensorflow
$ pip3 install keras
$ pip install opencv-python
```
## Entrenamiento
```sh
$ python3 train.py
```

## Evaluación
```sh
$ python3 train.py
```
La exactitud del modelo en el conjunto de evaluación es 0.9603.

## Predicción
```sh
$ python3 prediction.py
```
