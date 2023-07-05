from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical, plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from tensorflow.keras import backend as K
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import random
import cv2
import os
import glob
import cvlib as cv


def training():
    epochs = 100
    lr = 1e-3
    batch_size = 64
    img_dims = (96, 96, 3)

    data = []
    labels = []

    image_files = [f for f in glob.glob(
        r'./assets/gender_dataset_face' + "/**/*", recursive=True) if not os.path.isdir(f)]
    random.shuffle(image_files)

    for img in image_files:

        image = cv2.imread(img)

        image = cv2.resize(image, (img_dims[0], img_dims[1]))
        image = img_to_array(image)
        data.append(image)

        label = img.split(os.path.sep)[-2]
        if label == "Women":
            label = 1
        else:
            label = 0

        labels.append([label])

    data = np.array(data, dtype="float") / 255.0
    labels = np.array(labels)

    (trainX, testX, trainY, testY) = train_test_split(data, labels, test_size=0.2,
                                                      random_state=42)

    trainY = to_categorical(trainY, num_classes=2)
    testY = to_categorical(testY, num_classes=2)

    aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                             height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                             horizontal_flip=True, fill_mode="nearest")

    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(64, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(128, (3, 3), padding="same"))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))

        model.add(Flatten())
        model.add(Dense(1024))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("sigmoid"))

        return model

    model = build(width=img_dims[0], height=img_dims[1], depth=img_dims[2],
                  classes=2)

    model.compile(loss="binary_crossentropy",
                  optimizer='adam', metrics=["accuracy"])

    H = model.fit(aug.flow(trainX, trainY, batch_size=batch_size),
                  validation_data=(testX, testY),
                  steps_per_epoch=len(trainX) // batch_size,
                  epochs=epochs, verbose=1)

    model.save('gender_detection.h5')

    plt.style.use("ggplot")
    plt.figure()
    N = epochs
    plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
    plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
    plt.plot(np.arange(0, N), H.history["accuracy"], label="train_accuracy")
    plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_accuracy")

    plt.title("Training Loss and Accuracy")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss/accuracy")
    plt.legend(loc="upper right")

    plt.savefig('plot.png')


def start():
    model = load_model('gender_detection.h5')

    webcam = cv2.VideoCapture(0)

    width = 526
    height = 330
    webcam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    webcam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

    classes = ['man', 'woman']

    while webcam.isOpened():
        status, frame = webcam.read()

        face, confidence = cv.detect_face(frame)

        for idx, f in enumerate(face):

            (startX, startY) = f[0], f[1]
            (endX, endY) = f[2], f[3]

            cv2.rectangle(frame, (startX, startY),
                          (endX, endY), (0, 255, 0), 2)

            face_crop = np.copy(frame[startY:endY, startX:endX])

            face_crop = cv2.resize(face_crop, (96, 96))
            face_crop = face_crop.astype("float") / 255.0
            face_crop = img_to_array(face_crop)
            face_crop = np.expand_dims(face_crop, axis=0)

            conf = model.predict(face_crop)[0]

            idx = np.argmax(conf)
            label = classes[idx]

            label = "{}: {:.2f}%".format(label, conf[idx] * 100)

            Y = startY - 10 if startY - 10 > 10 else startY + 10

            cv2.putText(frame, label, (startX, Y),  cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        cv2.imshow("Tekan Q untuk Berhenti", frame)
        cv2.resizeWindow("Tekan Q untuk Berhenti", 526, 330)
        cv2.moveWindow("Tekan Q untuk Berhenti", 455, 150)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    webcam.release()
    cv2.destroyAllWindows()
