from keras.preprocessing.image import ImageDataGenerator
from keras import backend as K
from models import Trial_model5
from keras.callbacks import ModelCheckpoint, TensorBoard, CSVLogger
from keras import optimizers
from keras.metrics import Recall, Precision
import os

root_dir = './PlantVillage'
train_dir_src = os.path.join(root_dir, 'train') # path to training dataset after extracting
val_dir_src = os.path.join(root_dir, 'val') # path to validation dataset after extracting


img_height = 256
img_width = 256

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')


test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_dir_src,
    target_size=(img_width, img_height),
    batch_size=32,
    shuffle=True,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    val_dir_src,
    target_size=(img_width, img_height),
    batch_size=32,
    shuffle=True,
    class_mode='categorical')


model = Trial_model5()
model.compile(optimizer=optimizers.Adam(
    learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['acc', Recall(), Precision(), f1])

epochs = 30
batch_size = 32

checkpoint = ModelCheckpoint("best_model_trial5.hdf5", monitor='val_loss', verbose=1,
    save_best_only=True, mode='min', period=1)
logging = TensorBoard(log_dir='/content/logs')  
csv_logger = CSVLogger('/content/log_trial5.csv', append=True, separator=',')

# Train model on dataset
history = model.fit_generator(generator=train_generator,
                    validation_data=validation_generator,
                    use_multiprocessing=True,
                    epochs=epochs, 
                    verbose=1,
                    callbacks=[checkpoint, logging, csv_logger])

model.save_weights('final_epoch_trial5.h5')
