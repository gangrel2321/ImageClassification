import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import numpy as np
from keras.preprocessing.image import ImageDataGenerator, img_to_array, load_img
from keras.models import Sequential
from keras.layers import Dropout, Flatten, Dense
from keras import applications
from keras.utils.np_utils import to_categorical
import matplotlib.pyplot as plt
import math
#import cv2

#image dimensions
img_width, img_height = 256, 256

top_model_weights_path = 'bottleneck_fc_model.h5'
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'

#epochs to train
epochs = 20
#batch size for flow_from_directory and predict_generator
batch_size = 16

def save_bottleneck_features():
    #use pretrained VGG16 for feature extraction
    #include_top=False removes fully connected layers
    model = applications.VGG16(include_top=False, weights='imagenet')

    datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True
    )

    generator = datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width,img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
    )

    nb_train_samples = len(generator.filenames)
    print("Number of training samples:", nb_train_samples)
    num_classes = len(generator.class_indices)
    print("Number of Classes:", num_classes)

    predict_size_train = int(math.ceil(nb_train_samples / float(batch_size)))
    print("Predicting Train Data.")
    bottleneck_features_train = model.predict_generator(generator, predict_size_train)
    print("Data Predicted.")
    np.save('bottleneck_features_train_2.npy', bottleneck_features_train)
    generator = datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width,img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
    )

    nb_vali_samples = len(generator.filenames)

    predict_size_vali = int(math.ceil(nb_vali_samples / float(batch_size) ))
    print("Predicting Validation Data.")
    bottleneck_features_vali = model.predict_generator(generator, predict_size_vali)
    print("Data Predicted.")

    np.save('bottleneck_features_vali_2.npy', bottleneck_features_vali)

def train_top_model():

    ###########################################################################
    ######   Manipulate and Load Data from Directories and Bottleneck    ######
    ###########################################################################

    datagen_top = ImageDataGenerator(rescale = 1. / 255)
    generator_top = datagen_top.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = "categorical",
        shuffle = False
    )

    nb_train_samples = len(generator_top.filenames)
    num_classes = len(generator_top.class_indices)

    #save class indices to use in predictions, might remove
    np.save('class_indices.npy', generator_top.class_indices)

    #load features saved in bottleneck
    train_data = np.load('bottleneck_features_train_2.npy')

    #get class labels
    train_labels = generator_top.classes

    #convert to categorical vectors
    train_labels = to_categorical(train_labels, num_classes=num_classes)

    #now we do the same for validation , might make validation categorical
    generator_top = datagen_top.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = None,
        shuffle = False
    )

    nb_vali_samples = len(generator_top.filenames)

    #load features saved in bottleneck
    validation_data = np.load('bottleneck_features_vali_2.npy')

    validation_labels = generator_top.classes
    print("TEST:", validation_labels)
    validation_labels = to_categorical(validation_labels, num_classes= num_classes)

    ###########################################################################
    ###### Train Full-Connected Network (Top Model) w/ Bottleneck inputs ######
    ###########################################################################

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='sigmoid'))

    model.compile(optimizer='rmsprop',
        loss = 'categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(train_data, train_labels,
        epochs = epochs,
        batch_size = batch_size,
        validation_data = (validation_data, validation_labels)
        )

    model.save_weights(top_model_weights_path)

    (eval_loss, eval_accuracy) = model.evaluate(
        validation_data, validation_labels, batch_size=batch_size, verbose=1
    )

    print("[INFO] accuracy: {:.2f}%".format(eval_accuracy * 100))
    print("[INFO] Loss: {}".format(eval_loss))

    #plot results

    plt.figure(1)

    #accuracy summary
    plt.subplot(211)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    #loss summary

    plt.subplot(212)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
    plt.show()

#save_bottleneck_features()
train_top_model()
