import time
import numpy
from keras.datasets import mnist
from keras.models import Sequential
from keras.utils.np_utils import to_categorical
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.callbacks import EarlyStopping, ModelCheckpoint

numpy.random.seed(0)



model_name = "mnist.h5"
batch_size = 800
epochs = 64
patience = 8
validation_split = 0.2
optimizer = "sgd"
loss = "categorical_crossentropy"
metrics = ["accuracy"]




(x_train, y_train), _ = mnist.load_data()

indices = numpy.arange(y_train.shape[0])
numpy.random.shuffle(indices)
x_train = x_train[indices]
y_train = y_train[indices]

x_train = x_train / 255 # normalizacion
x_train = numpy.expand_dims(x_train, -1) # requerimiento de keras para poder operar
input_shape = x_train[0].shape # tamano de la entrada, en mnist es igual a (28, 28)

num_classes = len(set(y_train)) # numero de clases, en mnist es 10
y_train = to_categorical(y_train, num_classes = num_classes)






model = Sequential()

model.add(Conv2D(8, (3, 3), padding = "same", activation = "relu", input_shape = input_shape))
model.add(MaxPooling2D((2, 2), padding = "valid"))
model.add(Conv2D(8, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D((2, 2), padding = "valid"))
model.add(Conv2D(8, (3, 3), padding = "same", activation = "relu"))
model.add(MaxPooling2D((2, 2), padding = "valid"))
model.add(Flatten())
model.add(Dense(units = 64, activation = "relu"))
model.add(Dense(units = num_classes, activation = "softmax"))

model.compile(optimizer = optimizer, loss = loss, metrics = metrics)
model.summary()



earlystopper = EarlyStopping(patience = patience, verbose = 1)
checkpointer = ModelCheckpoint(model_name, verbose = 1, save_best_only = True)



print("epochs = ", epochs)
print("optimizer = ", optimizer)
print("loss = ", loss)
print("metrics = ", metrics)
print("patience = ", patience)
print("input_shape = ", input_shape)
print("validation_split = ", validation_split)

#################################### training

t = time.time()

model.fit(x = x_train, y = y_train, batch_size = batch_size, epochs = epochs, verbose = 1, callbacks = [earlystopper, checkpointer], validation_split = validation_split, shuffle = True)

print("elapsed time = ", (time.time() - t) / 3600, "hours")







## evaluación

#model = load_model("road.h5")

#(x_train, y_train), (x_test, y_test) = mnist.load_data()

#num_classes = len(set(y_train))

#x_test = numpy.reshape(x_test, (x_test.shape[0], -1))



#predictions = model.predict(x_test)
#y_hat = numpy.argmax(predictions, axis = -1)



#for x, y in zip(y_hat, y_test):
#	print(x, y)









