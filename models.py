from keras.layers import Input
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation, Flatten, Dense
import tensorflow as tf
from keras.models import Sequential


def Trial_model1():
    input11 = Input(shape=(256, 256, 3))
    maxp10 = MaxPooling2D(pool_size=(2, 2))(input11)
    conv11 = Conv2D(24, (5, 5), strides=(2, 2))(maxp10)
    bat11 = BatchNormalization()(conv11)
    Act11 = Activation("relu")(bat11)
    conv113 = Conv2D(16, (5, 5), strides=(1, 1))(Act11)
    bat113 = BatchNormalization()(conv113)
    Act12 = Activation("relu")(bat113)
    maxp11 = MaxPooling2D(pool_size=(2, 2))(Act12)
    conv12 = Conv2D(12, (3, 3))(maxp11)
    bat12 = BatchNormalization()(conv12)
    Act13 = Activation("relu")(bat12)
    conv13 = Conv2D(8, (3, 3))(Act13)
    bat13 = BatchNormalization()(conv13)
    Act14 = Activation("relu")(bat13)
    conv14 = Conv2D(8, (3, 3))(Act14)
    bat14 = BatchNormalization()(conv14)
    Act15 = Activation("relu")(bat14)
    maxp14 = MaxPooling2D(pool_size=(2, 2))(Act15)
    flat11 = Flatten()(maxp14)
    dense14 = Dense(38, activation='softmax')(flat11)

    return tf.keras.Model(inputs=input11, outputs=dense14)


def Trial_model2():
    input11 = Input(shape=(256, 256, 3))
    maxp10 = MaxPooling2D(pool_size=(2, 2))(input11)
    conv11 = Conv2D(24, (5, 5), strides=(2, 2))(maxp10)
    bat11 = BatchNormalization()(conv11)
    Act11 = Activation("relu")(bat11)
    conv113 = Conv2D(16, (5, 5), strides=(1, 1))(Act11)
    bat113 = BatchNormalization()(conv113)
    Act12 = Activation("relu")(bat113)
    maxp11 = MaxPooling2D(pool_size=(2, 2))(Act12)
    conv12 = Conv2D(12, (3, 3))(maxp11)
    bat12 = BatchNormalization()(conv12)
    Act13 = Activation("relu")(bat12)
    conv13 = Conv2D(8, (3, 3))(Act13)
    bat13 = BatchNormalization()(conv13)
    Act14 = Activation("relu")(bat13)
    conv14 = Conv2D(8, (3, 3))(Act14)
    bat14 = BatchNormalization()(conv14)
    Act15 = Activation("relu")(bat14)
    maxp14 = MaxPooling2D(pool_size=(2, 2))(Act15)
    flat11 = Flatten()(maxp14)
    dense14 = Dense(38, activation='softmax')(flat11)

    return tf.keras.Model(inputs=input11, outputs=dense14)


def Trial_model3():
    input11 = Input(shape=(256,256,3))
    maxp10 = MaxPooling2D(pool_size=(2,2))(input11)
    conv11 = Conv2D(4,(3,3),strides=(2,2))(maxp10)
    bat11 = BatchNormalization()(conv11)
    Act11 = Activation("relu")(bat11)
    conv113 = Conv2D(8,(3,3),strides=(1,1))(Act11)
    bat113 = BatchNormalization()(conv113)
    Act12 = Activation("relu")(bat113)
    maxp11 = MaxPooling2D(pool_size=(2,2))(Act12)
    conv12 = Conv2D(12,(5,5))(maxp11)
    bat12 = BatchNormalization()(conv12)
    Act13 = Activation("relu")(bat12)
    conv13 = Conv2D(16,(5,5))(Act13)
    bat13 = BatchNormalization()(conv13)
    Act14 = Activation("relu")(bat13)
    conv14 = Conv2D(32,(5,5))(Act14)
    bat14 = BatchNormalization()(conv14)
    Act15 = Activation("relu")(bat14)
    maxp14 = MaxPooling2D(pool_size=(2,2))(Act15)
    flat11 = Flatten()(maxp14)
    dense14 = Dense(38,activation='softmax')(flat11)

    return tf.keras.Model(inputs=input11,outputs=dense14)


def Trial_model4():
    input11 = Input(shape=(256, 256, 3))
    maxp10 = MaxPooling2D(pool_size=(2, 2))(input11)
    conv11 = Conv2D(4, (2, 2), strides=(2, 2))(maxp10)
    bat11 = BatchNormalization()(conv11)
    Act11 = Activation("relu")(bat11)
    conv113 = Conv2D(8, (3, 3), strides=(1, 1))(Act11)
    bat113 = BatchNormalization()(conv113)
    Act12 = Activation("relu")(bat113)
    maxp11 = MaxPooling2D(pool_size=(2, 2))(Act12)
    conv12 = Conv2D(12, (3, 3))(maxp11)
    bat12 = BatchNormalization()(conv12)
    Act13 = Activation("relu")(bat12)
    conv13 = Conv2D(12, (3, 3))(Act13)
    bat13 = BatchNormalization()(conv13)
    Act14 = Activation("relu")(bat13)
    conv14 = Conv2D(16, (3, 3))(Act14)
    bat14 = BatchNormalization()(conv14)
    Act15 = Activation("relu")(bat14)
    maxp14 = MaxPooling2D(pool_size=(2, 2))(Act15)
    conv16 = Conv2D(32, (3, 3))(maxp14)
    bat16 = BatchNormalization()(conv16)
    Act16 = Activation("relu")(bat16)
    flat11 = Flatten()(Act16)
    dense14 = Dense(38, activation='softmax')(flat11)

    return tf.keras.Model(inputs=input11, outputs=dense14)


def Trial_model5():
    input11=Input(shape=(256,256,3))
    maxp10 = MaxPooling2D(pool_size=(2,2))(input11)
    conv11 = Conv2D(4,(2,2),strides=(2,2))(maxp10)
    bat11 = BatchNormalization()(conv11)
    Act11 = Activation("relu")(bat11)
    conv113 = Conv2D(8,(3,3),strides=(1,1))(Act11)
    bat113 = BatchNormalization()(conv113)
    Act12 = Activation("relu")(bat113)
    maxp11 = MaxPooling2D(pool_size=(2,2))(Act12)
    conv12 = Conv2D(12,(3,3))(maxp11)
    bat12 = BatchNormalization()(conv12)
    Act13 = Activation("relu")(bat12)
    conv13 = Conv2D(16,(3,3))(Act13)
    bat13 = BatchNormalization()(conv13)
    Act14 = Activation("relu")(bat13)
    conv14 = Conv2D(32,(3,3))(Act14)
    bat14 = BatchNormalization()(conv14)
    Act15 = Activation("relu")(bat14)
    maxp14 = MaxPooling2D(pool_size=(2,2))(Act15)
    conv16 = Conv2D(64,(3,3))(maxp14)
    bat16 = BatchNormalization()(conv16)
    Act16 = Activation("relu")(bat16)
    flat11 = Flatten()(Act16)
    dense14 = Dense(38,activation='softmax')(flat11)

    return tf.keras.Model(inputs=input11,outputs=dense14)


def Trial_model6():
    input11 = Input(shape=(256, 256, 3))
    maxp10 = MaxPooling2D(pool_size=(2, 2))(input11)
    conv11 = Conv2D(4, (5, 5), strides=(2, 2))(maxp10)
    bat11 = BatchNormalization()(conv11)
    Act11 = Activation("relu")(bat11)
    conv113 = Conv2D(8, (5, 5), strides=(1, 1))(Act11)
    bat113 = BatchNormalization()(conv113)
    Act12 = Activation("relu")(bat113)
    maxp11 = MaxPooling2D(pool_size=(2, 2))(Act12)
    conv12 = Conv2D(12, (3, 3))(maxp11)
    bat12 = BatchNormalization()(conv12)
    Act13 = Activation("relu")(bat12)
    conv13 = Conv2D(16, (3, 3))(Act13)
    bat13 = BatchNormalization()(conv13)
    Act14 = Activation("relu")(bat13)
    maxp12 = MaxPooling2D(pool_size=(2, 2))(Act14)
    conv14 = Conv2D(32, (3, 3))(maxp12)
    bat14 = BatchNormalization()(conv14)
    Act15 = Activation("relu")(bat14)
    maxp14 = MaxPooling2D(pool_size=(2, 2))(Act15)
    flat11 = Flatten()(maxp14)
    dense11 = Dense(64, activation='relu')(flat11)
    dense14 = Dense(38, activation='softmax')(dense11)

    return tf.keras.Model(inputs=input11, outputs=dense14)


def VGG19():
    conv_base = VGG19(weights=None,
                  include_top=False,
                  input_shape=(256,256,3))

    conv_base.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(38, activation='softmax'))

    return model

def ResNet50():
    conv_base =ResNet50(weights='imagenet',
                  include_top=False,
                  input_shape=(256,256,3))
    conv_base.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(38, activation='softmax'))

    return model

def Xception():
    conv_base = Xception(weights='imagenet',
                  include_top=False,
                  input_shape=(256,256,3))
    conv_base.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(38, activation='softmax'))

    return model

def MobileNetV2():
    conv_base = MobileNetV2(weights='imagenet',
                  include_top=False,
                  input_shape=(256,256,3))
    conv_base.trainable = False
    model = Sequential()
    model.add(conv_base)
    model.add(Flatten())
    model.add(Dense(38, activation='softmax'))

    return model
