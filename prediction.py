import time
import numpy
import cv2
from keras.models import load_model






model_name = "mnist.h5"
model = load_model(model_name)

cap = cv2.VideoCapture(0)
_, frame = cap.read()
h, w, c = frame.shape
crop_size = 150

padx = (w - crop_size) // 2
pady = (h - crop_size) // 2






while(True):
	
	ret, frame = cap.read() # Capture frame-by-frame
	
	label = numpy.zeros((crop_size // 5, 2 * crop_size), dtype = numpy.uint8)
	
	frame = frame[pady:-pady, padx:-padx, :]
	
	image_0 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # Convert to grayscale
	
	_, image_1 = cv2.threshold(image_0, 100, 255, cv2.THRESH_BINARY) # Convert to black and white
	image_1 = cv2.bitwise_not(image_1) # Invert colors
	
	test = cv2.resize(image_1, (28, 28), interpolation = cv2.INTER_AREA) # Resize to mnist size
	test = test / 255 # normalize
	
	test = numpy.expand_dims(test, -1) # Neccesary to fit Keras model
	test = numpy.expand_dims(test, 0)
	
	probs = model.predict(test) # prediction
	prediction = numpy.argmax(probs)
	
	cv2.putText(image_0, "camera", (12, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
	cv2.putText(image_1, "cnn input", (12, 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA)
	
	# Display the resulting frame
	image_h = cv2.hconcat([image_0, image_1])
	
	cv2.putText(label, "label: " + str(prediction), (10, crop_size // 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, cv2.LINE_AA)
	image_v = cv2.vconcat([image_h, label])
	
	cv2.imshow('numbers', image_v)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
