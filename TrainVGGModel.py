import os
import cv2
import Utils
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Dropout, GlobalAveragePooling2D, Input, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model
from keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Loading augmented dataset
label_map = {'no': 0, 'yes': 1} # used to map 0 for negatives and 1 for positives
images, labels = Utils.load_images_augmented(label_map)
print("Shape of images:", images.shape)
print("Shape of labels:", labels.shape)

# Cropping brain area
cropped_images = []
for img in images:
    cropped_img = Utils.crop_brain_region(img, (224, 224))
    cropped_images.append(cropped_img)
cropped_images = np.array(cropped_images)

# Splitting data onto train and test
X_train, x_test, Y_train, y_test = train_test_split(cropped_images, labels, test_size=0.2, shuffle=True)
X_train_scaled=X_train/255
X_test_scaled=x_test/255

# Loading base VGG16 model and adjusting settings
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers[:-5]:
    layer.trainable = False
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(1, activation='sigmoid')(x)
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(loss='binary_crossentropy', optimizer=RMSprop(learning_rate=1e-4), metrics=['accuracy'])

# Setting training parametres
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-6)
history = model.fit(
    X_train_scaled,
    Y_train,
    epochs=200,
    validation_split=0.2,
    callbacks=[early_stopping, reduce_lr]
)
# Saving model
model.save('Brain_MRI_VGG16_Augmented.keras')

# Showing graph of loss change on every epoch
plt.plot(history.history['loss'], label='Потеря обучения')
plt.plot(history.history['val_loss'], label='Потеря валидации')
plt.xlabel('Эпохи')
plt.ylabel('Функция потери')
plt.legend()
plt.show()
# Showing graph of accuracy change on every epoch
plt.plot(history.history['accuracy'], label='Точность на обучающей выборке')
plt.plot(history.history['val_accuracy'], label='Точность на валидационной выборке')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()
