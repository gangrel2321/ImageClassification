from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
import numpy as np
import sys

inputImg = sys.argv[1]
# rModel = get model
img = image.load_img(inputImg, target_size=(224,224))
x = image.img_to_array(img)
x = np.expand_dims(x,axis=0)
x = preprocess_input(x)

#get graph hierarchy
k_graph = GRAPH_HIERARCHY
neural_pred = rModel.predict(x)
#initialize probabilites for each node using neural network
for i in k_graph:
    k_graph.prob = neural_pred[i]


print('Results: ', decode_predictions(preds, top=10)[0])
