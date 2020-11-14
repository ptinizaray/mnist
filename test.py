import time
import numpy
from keras.datasets import mnist
from keras.models import load_model



model_name = "mnist.h5"
model = load_model(model_name)
model.summary()

_, (x_test, y_test)= mnist.load_data()

num_classes = len(set(y_test)) # en mnist existen 10 clases

x_test = numpy.expand_dims(x_test, -1)
predictions = model.predict(x_test)
y_hat = numpy.argmax(predictions, axis = -1)

acc = 0
for p in range(num_classes):
	acc += numpy.sum(numpy.logical_and(y_test == p, y_hat == p))

accuracy = acc / y_test.shape[0]

print(f"accuracy = {accuracy}")


cm = numpy.zeros((num_classes, num_classes), dtype = numpy.uint32)

for p in range(num_classes):
	for q in range(num_classes):
		cm[p, q] += numpy.sum(numpy.logical_and(y_test == p, y_hat == q))

norm = numpy.zeros((num_classes, num_classes), dtype = numpy.float16)
s = numpy.sum(cm, axis = 0)
for i in range(num_classes):
	norm[:, i] = numpy.around(cm[:, i] / s[i], 3)
print(f"confusion_matrix =\n{norm}")
