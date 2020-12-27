# imports
import matplotlib.pyplot as plt
import PIL
import tensorflow as tf
import numpy as np
import os
import json
from tensorflow.python.keras.models import Model, Sequential, load_model
from tensorflow.python.keras.layers import Dense, Flatten, Dropout,GlobalAveragePooling2D,Input
from tensorflow.python.keras.applications import VGG16
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizers import Adam, RMSprop, Adadelta


def finetune_vgg16_model(base_model, transfer_layer, x_trainable, dropout, fc_layers, num_classes, new_weights = ""):
    # number of freezed layers: 0 (by a default)
    freezed_layers = 0
    if x_trainable != "all":
        if x_trainable == 0:
            for layer in base_model.layers:
                layer.trainable = False
                freezed_layers = freezed_layers + 1
        else:
            for layer in base_model.layers[:-x_trainable]:
                layer.trainable = False
                freezed_layers = freezed_layers + 1  
    all_layers = len(base_model.layers)
    print("Number of all layers in a feature-extractor part of model: {}.".format(all_layers))
    print("Number of freezed (untrainable) layers in a feature-extractor part of model: {}.".format(freeze))
    # adding custom layers to the classification part of a model
    x = transfer_layer.output
    #x = Dense(num_classes, activation='softmax')
    x = Flatten()(x)
    for fc in fc_layers:
        # New FC layer, random init
        x = Dense(fc, activation='relu')(x) 
        x = Dropout(dropout)(x)
    # New softmax layer
    predictions = Dense(num_classes, activation='softmax')(x)  
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights != "":
        finetune_model.load_weights(new_weights)
    return finetune_model


def finetune_resnet50_model(base_model, transfer_layer, x_trainable, dropout, fc_layers, num_classes, new_weights = ""):
    # number of freezed layers: 0 (by a default)
    freezed_layers = 0
    if x_trainable != "all":
        if x_trainable == 0:
            for layer in base_model.layers:
                layer.trainable = False
                freezed_layers = freezed_layers + 1
        else:
            for layer in base_model.layers[:-x_trainable]:
                layer.trainable = False
                freezed_layers = freezed_layers + 1      
    all_layers = len(base_model.layers)
    print("Number of all layers in a feature-extractor part of model: {}.".format(all_layers))
    print("Number of freezed (untrainable) layers in a feature-extractor part of model: {}.".format(freeze))
    # adding custom layers to the classification part of a model
    x = transfer_layer.output
    x = GlobalAveragePooling2D()(x)
    predictions = Dense(num_classes, activation='softmax')(x)  
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights != "":
        finetune_model.load_weights(new_weights)
    return finetune_model


def finetune_inceptionv3(base_model, transfer_layer, x_trainable, dropout, fc_layers, num_classes, new_weights = ""):
    # number of freezed layers: 0 (by a default)
    freezed_layers = 0
    if x_trainable != "all":
        if x_trainable == 0:
            for layer in base_model.layers:
                layer.trainable = False
                freezed_layers = freezed_layers + 1
        else:
            for layer in base_model.layers[:-x_trainable]:
                layer.trainable = False
                freezed_layers = freezed_layers + 1    
    all_layers = len(base_model.layers)
    print("Number of all layers in a feature-extractor part of model: {}.".format(all_layers))
    print("Number of freezed (untrainable) layers in a feature-extractor part of model: {}.".format(freezed_layers))
    # adding custom layers to the classification part of a model
    x = transfer_layer.output
    x = GlobalAveragePooling2D(name='avg_pool')(x)
    x = Dropout(dropout)(x)
    predictions = Dense(num_classes, activation='softmax')(x)  
    finetune_model = Model(inputs=base_model.input, outputs=predictions)
    if new_weights != "":
        finetune_model.load_weights(new_weights)
    return finetune_model


if __name__ == "__main__":
    base_model = InceptionV3(include_top=False, weights = 'imagenet', input_shape=(299,299,3))
    input_shape = base_model.layers[0].output_shape[1:3]
    transfer_layer = base_model.get_layer(index=-1)
    print(transfer_layer)                
    new_model = finetune_inceptionv3(base_model, transfer_layer, "all", 0.5, [1024, 1024], 196)
    optimizer = Adam(lr=0.000001)
    #new_model.load_weights('../saved_models/20190530_1112/weights.best.hdf5')
    new_model.compile(optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    print(new_model.summary())
    for layer in new_model.layers: print(layer, layer.trainable)
