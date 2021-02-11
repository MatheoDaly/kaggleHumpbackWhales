import csv
import numpy as np
import pandas as pd
import os
import gc
from numpy import array, argmax
from IPython.display import Image
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
import keras

from tqdm import tqdm
from keras.models import Model
from keras import optimizers

from PIL import Image, ImageFilter, ImageOps


PATH = "/home/data/challenge_deep"

# load the training data
trainData = pd.read_csv(PATH + "/train.csv")


def zoom_at(img, zoom, x=100, y=100):
    w, h = img.size
    size = (200, 200)
    img = img.resize((round(w * zoom), round(w * zoom)))

    final_image = Image.new(mode='RGB', size=size, color=(127, 127, 127))
    final_image.paste(img, (0, 0))

    return final_image


def prepareImages(data, m, dataset, image_size=200):
    X_train = np.zeros((m, image_size, image_size))

    count = 0

    for fig in tqdm(data['Image']):
        if count < m:
            # load images
            img = Image.open(dataset + "/" + fig)

            ## FILTRES

            # Resize
            img = img.resize((200,200))
            # Flou bilatéral
            img = img.filter(ImageFilter.SMOOTH)
            # Netteté
            img = img.filter(ImageFilter.SHARPEN)
            # Embossage
            img = img.filter(ImageFilter.EMBOSS)
            # Netteté
            img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=130, threshold=2))

            # ## DATA AUGMENTATION

            # # Zoom aleatory
            # zoom_size = np.random.uniform(0.8, 1.2)
            # img = zoom_at(img, zoom_size)

            # # Rotation aleatory
            # angle = random.randint(-20, 20)
            # img = img.rotate(angle, fillcolor='grey')
            # img = img.filter(ImageFilter.SHARPEN)

            # Noir et blanc, et enregistrement
            img = ImageOps.grayscale(img)

            # Enregistrement dans X_train
            img = keras.applications.resnet.preprocess_input(np.array(img))
            X_train[count] = img

            # Suivi
            count += 1

    count = 0

    return X_train


def prepareY(Y):
    values = array(Y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = onehot_encoded
    return y, label_encoder


def prepareY2(Y):
    values = array(Y)
    label_encoder = LabelEncoder()
    integer_encoded = label_encoder.fit_transform(values)

    onehot_encoder = OneHotEncoder(sparse=False)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    onehot_encoded = onehot_encoder.fit_transform(integer_encoded)

    y = integer_encoded
    return y, label_encoder


# Premiere initialisation des Y et des X
Y = trainData['Id']
y_onehot, label_encoder = prepareY(Y)
y_integer, label_encoder = prepareY2(Y)

Y_train_onehot = y_onehot
Y_train_integer = y_integer

X_train = prepareImages(trainData, 9850, PATH + "/train")

X_train = np.repeat(X_train[..., np.newaxis], 3, -1)
print("INIT X_TRAIN AND Y_TRAIN")
# BOUCLE
#for i in tqdm(range(9)):
    # Preparation des images
#    X = prepareImages(trainData, 9850, PATH + "/train")
#    X = np.repeat(X[..., np.newaxis], 3, -1)
    # Concatenation des images
#    X_train = np.concatenate((X_train, X))

    # Concatenation des Y
#    Y_train_onehot = np.concatenate((Y_train_onehot, y_onehot))
#    Y_train_integer = np.concatenate((Y_train_integer, y_integer))

print("X_TRAIN AND Y_TRAIN PROCESSED")
baseModel = keras.applications.ResNet50(weights="imagenet", include_top=False,
                                        input_tensor=keras.layers.Input(shape=(200, 200, 3)))
for layer in baseModel.layers:
    layer.trainable = False
mod = baseModel.output

mod = keras.layers.BatchNormalization(axis=3, name='bn0')(mod)
mod = keras.layers.Activation('relu')(mod)

mod = keras.layers.MaxPooling2D((2, 2), name='out_max_pool')(mod)
mod = keras.layers.Conv2D(64, (2, 2), strides=(1, 1), name="out_conv1")(mod)
mod = keras.layers.Activation('relu')(mod)
mod = keras.layers.AveragePooling2D((1, 1), name='out_avg_pool')(mod)

mod = keras.layers.Flatten()(mod)
mod = keras.layers.Dense(500, activation="relu", name='out_relu')(mod)
mod = keras.layers.Dropout(0.8)(mod)
mod = keras.layers.Dense(4251, activation='softmax', name='out_softmax')(mod)

mod = Model(inputs=baseModel.input, outputs=mod)
opt = optimizers.Adam(lr=0.01, beta_1=0.9, beta_2=0.999, epsilon=0.0000001, decay=0.0, amsgrad=False)

mod.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
print("NEURAL NETWORK INITIALIZED")

history = mod.fit(X_train, Y_train_onehot, epochs=100, batch_size=100, verbose=1)
mod.save("model.hdf5")
print("NEURAL NETWORK TRAINED")
# open test data
test = os.listdir(PATH + "/test/")

print("BEGINNING TEST")
# separate data into different DataFrames due to memory constraints
col = ['Image']
testData1 = pd.DataFrame(test[0:3899], columns=col)
testData2 = pd.DataFrame(test[3900:7799], columns=col)
testData3 = pd.DataFrame(test[7800:11699], columns=col)
testData4 = pd.DataFrame(test[11700:15609], columns=col)
testData = pd.DataFrame(test, columns=col)

# X_test = prepareImages(testData1, 15610, "test")
gc.collect()
X = prepareImages(testData1, 3900, PATH + "/test")
X /= 255

X = np.repeat(X[..., np.newaxis], 3, -1)
predictions1 = mod.predict(np.array(X), verbose=1)
gc.collect()

X = prepareImages(testData2, 3900, PATH + "/test")
X /= 255
X = np.repeat(X[..., np.newaxis], 3, -1)
predictions2 = mod.predict(np.array(X), verbose=1)
gc.collect()
X = prepareImages(testData3, 3900, PATH + "/test")
X /= 255
X = np.repeat(X[..., np.newaxis], 3, -1)
predictions3 = mod.predict(np.array(X), verbose=1)
gc.collect()
X = prepareImages(testData4, 3910, PATH + "/test")
X /= 255
X = np.repeat(X[..., np.newaxis], 3, -1)
predictions4 = mod.predict(np.array(X), verbose=1)
gc.collect()
print("TEST IMAGES PROCESSED")
# concatenate all the predictions in the same vector
print("PREPARING PREDICTIONS")
predictions = np.concatenate((predictions1, predictions2), axis=0)
predictions = np.concatenate((predictions, predictions3), axis=0)
predictions = np.concatenate((predictions, predictions4), axis=0)
gc.collect()
print("PREDICTIONS PROCESSED")
# choose predictions with highest probability. For each value I choose, I set the probability to zero, so it can't be picked again.


copy_pred = np.copy(predictions)
idx = np.argmax(copy_pred, axis=1)
copy_pred[:, idx] = 0
idx2 = np.argmax(copy_pred, axis=1)
copy_pred[:, idx2] = 0
idx3 = np.argmax(copy_pred, axis=1)
copy_pred[:, idx3] = 0
idx4 = np.argmax(copy_pred, axis=1)
copy_pred[:, idx4] = 0
idx5 = np.argmax(copy_pred, axis=1)

# convert the one-hot vectors to their names
results = []

print(idx[0:10])
print(idx2[0:10])
print(idx3[0:10])
print(idx4[0:10])
print(idx5[0:10])
threshold = 0.05  # threshold - only consider answers with a probability higher than it
for i in range(0, predictions.shape[0]):
    # for i in range(0, 10):
    each = np.zeros((4251, 1))
    each2 = np.zeros((4251, 1))
    each3 = np.zeros((4251, 1))
    each4 = np.zeros((4251, 1))
    each5 = np.zeros((4251, 1))
    if ((predictions[i, idx5[i]] > threshold)):
        each5[idx5[i]] = 1
        each4[idx4[i]] = 1
        each3[idx3[i]] = 1
        each2[idx2[i]] = 1
        each[idx[i]] = 1
        tags = [label_encoder.inverse_transform([argmax(each)])[0], label_encoder.inverse_transform([argmax(each2)])[0],
                label_encoder.inverse_transform([argmax(each3)])[0],
                label_encoder.inverse_transform([argmax(each4)])[0],
                label_encoder.inverse_transform([argmax(each5)])[0]]
    else:
        if ((predictions[i, idx4[i]] > threshold)):
            print(predictions[i, idx4[i]])
            each4[idx4[i]] = 1
            each3[idx3[i]] = 1
            each2[idx2[i]] = 1
            each[idx[i]] = 1
            tags = [label_encoder.inverse_transform([argmax(each)])[0],
                    label_encoder.inverse_transform([argmax(each2)])[0],
                    label_encoder.inverse_transform([argmax(each3)])[0],
                    label_encoder.inverse_transform([argmax(each4)])[0]]
        else:
            if ((predictions[i, idx3[i]] > threshold)):
                each3[idx3[i]] = 1
                each2[idx2[i]] = 1
                each[idx[i]] = 1
                tags = [label_encoder.inverse_transform([argmax(each)])[0],
                        label_encoder.inverse_transform([argmax(each2)])[0],
                        label_encoder.inverse_transform([argmax(each3)])[0]]
            else:
                if ((predictions[i, idx2[i]] > threshold)):
                    each2[idx2[i]] = 1
                    each[idx[i]] = 1
                    tags = [label_encoder.inverse_transform([argmax(each)])[0],
                            label_encoder.inverse_transform([argmax(each2)])[0]]
                else:
                    each[idx[i]] = 1
                    tags = label_encoder.inverse_transform([argmax(each)])[0]
    results.append(tags)

# write the predictions in a file to be submitted in the competition.
myfile = open('output.csv', 'w')

column = ['Image', 'Id']

wrtr = csv.writer(myfile, delimiter=',')
wrtr.writerow(column)

for i in range(0, testData.shape[0]):
    pred = ""
    if (len(results[i]) == 5):
        if (results[i][4] != results[i][0]):
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3] + " " + results[i][4]
        else:
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3]
    else:
        if (len(results[i]) == 4):
            pred = results[i][0] + " " + results[i][1] + " " + results[i][2] + " " + results[i][3]
        else:
            if (len(results[i]) == 3):
                pred = results[i][0] + " " + results[i][1] + " " + results[i][2]
            else:
                if (len(results[i]) == 2):
                    pred = results[i][0] + " " + results[i][1]
                else:
                    pred = results[i]

    result = [testData['Image'][i], pred]
    # print(result)
    wrtr.writerow(result)

myfile.close()
