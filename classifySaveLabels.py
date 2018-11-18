import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# import the necessary packages
from sklearn.preprocessing import MultiLabelBinarizer
from keras.preprocessing.image import img_to_array
from NN_pagerank import pageRank, toMatrix, naive
from sklearn.metrics import accuracy_score
from keras.models import load_model
from collections import defaultdict
import matplotlib.pyplot as plt
from keras import applications
from drawnow import drawnow
from imutils import paths
import numpy as np
import argparse
#import imutils
import pickle
import random
import cv2

dataSize = 1638 #size of test set

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model")
ap.add_argument("-d", "--directory", required=True,
	help="path to test images")
ap.add_argument("-l", "--LabelBin", required=False,
	help="path to MultiLabelBinarizer")
args = vars(ap.parse_args())

nnLabels = defaultdict(lambda: []) #label probabilities returned by NN
pageLabels = defaultdict(lambda: []) #label probabilities returned by pageRank

# initialize the data and labels
test_labels = []
testImagePaths = sorted(list(paths.list_images(args["directory"])))
# loop over the input images
for imagePath in testImagePaths:
    l = label = imagePath.split(os.path.sep)[-2].split("_")
    test_labels.append(l)

test_labels = np.array(test_labels)
if not (args["LabelBin"] == None):
	mlb =
mlb = MultiLabelBinarizer()
test_labels = mlb.fit_transform(test_labels)

"""#save labels so we don't need to recompute them
print("[INFO] serializing label binarizer...")
f = open("TestSetBinarizer.pickle", "wb")
f.write(pickle.dumps(mlb))
f.close()
print("Done")
"""
# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
beg_model = applications.VGG16(include_top=False, weights='imagenet')
end_model = load_model(args["model"])
#mlb = pickle.loads(open(args["labelbin"], "rb").read())

total = 0
nn_acc = 0
page_acc = 0
naive_acc = 0
total_nn_acc = [None]*dataSize #compound neural network accuracy over time
time_nn_acc = [None]*dataSize #accuracy for every image
total_page_acc = [None]*dataSize #compound pageRank accuracy over time
time_page_acc = [None]*dataSize #accuracy for every image
# classify the input image then find the indexes of the two class
# labels with the *largest* probability
#print(len(testImagePaths))

for imagePath in testImagePaths:
    ext = imagePath.split(".")[-1]
    ext = ext.lower()

    #if ext != "jpg" and ext != "png" and ext != "jpeg":
    #    continue

    # load the image
    im = cv2.imread(imagePath)

    # pre-process the image for classification
    im = cv2.resize(im, (256, 256))
    im = im.astype("float") / 255.0
    im = img_to_array(im)
    im = np.expand_dims(im, axis=0)

    image = imagePath.split("\\")[-1]

    #print(image)

    probs = end_model.predict(beg_model.predict(im))[0]
    nnLabels[image] = probs
    #print(probs)

    surfRate = 0.6
    pageProbs = pageRank(toMatrix(probs), surfRate) #0.2 seemed to work well at first
    pageLabels[image] = pageProbs
    #print(pageProbs)

    naiveProbs = naive(probs)

    c_nn_acc = accuracy_score(test_labels[total:total+1][0], list(map(lambda x: int(x + 0.49999),probs)) )
    #nn_acc += c_nn_acc
    #total_nn_acc[total] = nn_acc / total
    time_nn_acc[total] = c_nn_acc

    c_page_acc = accuracy_score(test_labels[total:total+1][0], list(map(lambda x: int(x + 0.49999),pageProbs)) )
    page_acc += c_page_acc
    total_page_acc[total] = page_acc / total
    time_page_acc[total] = c_page_acc

    #c_naive_acc = accuracy_score(test_labels[total:total+1][0], list(map(lambda x: int(x + 0.49999),naiveProbs)) )
    #naive_acc += c_naive_acc
    total += 1

    #print progress
    print(imagePath.split("\\")[-2])
    print("NN: ", nn_acc / total)
    print("Pg: ", page_acc / total)
    #print("Nv: ", naive_acc / total)

#nn_acc /= total
page_acc /= total
#print("NN Accuracy:", nn_acc)
print("PageRank Accuracy:", page_acc)

#save results
#np.save("./output/final_CNN_acc_test" , nn_acc )
nn_acc = np.load("./output/final_CNN_acc_test.npy")
np.save("./output/final_page_acc_test" , page_acc )
#np.save("./output/total_nn_acc" , total_nn_acc )
total_nn_ac = np.load("./output/total_nn_acc.npy")
np.save("./output/total_page_acc" , total_page_acc )
np.save("./output/temporal_nn_acc" , time_nn_acc )
#time_nn_acc = np.load("./output/temporal_nn_acc")
np.save("./output/temporal_page_acc" , time_page_acc )

#Plot results

#Compound accuracy
plt.figure(1)
plt.subplot(211)
plt.xlim(0,dataSize)
plt.ylim(0.6,1)
plt.plot(total_nn_acc)
plt.plot(total_page_acc)
plt.title('Compound Accuracy Over Time')
plt.ylabel('Accuracy')
plt.xlabel('Image')
plt.legend(['CNN', 'PageRank'], loc='upper left')
plt.savefig('./output/CompoundAccuracyTest_60.png', dpi=500)

#temporal accuracy
plt.figure(2)
plt.subplot(211)
plt.xlim(0,dataSize)
plt.ylim(0,1)
plt.scatter(list(range(dataSize)), time_nn_acc)
plt.scatter(list(range(dataSize)), time_page_acc)
plt.title('Individual Accuracy Over Time')
plt.ylabel('Accuracy')
plt.xlabel('Image')
plt.legend(['CNN', 'PageRank'], loc='upper left')
plt.savefig('./output/TemporalAccuracyTest_60.png', dpi=500)
plt.show()
