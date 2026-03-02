import os
import Utils
from sklearn.model_selection import train_test_split
import numpy as np
import keras
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense,Dropout, GlobalAveragePooling2D, Input, Conv2D, Flatten, MaxPooling2D, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from keras.optimizers import RMSprop, Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau


# Loading augmented dataset
label_map = {'no': 0, 'yes': 1} # used to map 0 for negatives and 1 for positives
images, labels = Utils.load_images_augmented(label_map)
print("Shape of images:", images.shape)
print("Shape of labels:", labels.shape)

# Showing of different image augmentations
plt.figure(figsize=(15, 5))
plt.subplot(3, 3, 1)
plt.imshow(images[labels == 1][0])  # Filter images with label 'yes'
plt.title("Оригинал")
plt.axis('off')
plt.subplot(3, 3, 2)
plt.imshow(images[labels == 1][1])  # Filter images with label 'yes'
plt.title("Поворот на 90° по часовой стрелке")
plt.axis('off')
plt.subplot(3, 3, 3)
plt.imshow(images[labels == 1][2])  # Filter images with label 'yes'
plt.title("Поворот на 90° против часовой стрелки")
plt.axis('off')
plt.subplot(3, 3, 4)
plt.imshow(images[labels == 1][3])  # Filter images with label 'yes'
plt.title("Отзеркаленное по горизонтали")
plt.axis('off')
plt.subplot(3, 3, 5)
plt.imshow(images[labels == 1][4])  # Filter images with label 'yes'
plt.title("Отзеркаленное по вертикали")
plt.axis('off')
plt.subplot(3, 3, 6)
plt.imshow(images[labels == 1][5])  # Filter images with label 'yes'
plt.title("Изображение с повышенной яркостью")
plt.axis('off')
plt.subplot(3, 3, 7)
plt.imshow(images[labels == 1][6])  # Filter images with label 'yes'
plt.title("Изображение с пониженной яркостью")
plt.axis('off')
plt.subplot(3, 3, 8)
plt.imshow(images[labels == 1][7])  # Filter images with label 'yes'
plt.title("Изображение с высокой контрастностью")
plt.axis('off')
plt.subplot(3, 3, 9)
plt.imshow(images[labels == 1][8])  # Filter images with label 'yes'
plt.title("Изображение с низкой контрастностью")
plt.axis('off')
plt.tight_layout()
plt.show()

# Crop brain area from images
cropped_images = []
for img in images:
    cropped_img = Utils.crop_brain_region(img, (224, 224))
    cropped_images.append(cropped_img)
cropped_images = np.array(cropped_images)

# Splitting data onto train and test
X_train, x_test, Y_train, y_test = train_test_split(cropped_images, labels, test_size=0.2, shuffle=True)
X_train_scaled=X_train/255
X_test_scaled=x_test/255

# Defining model architecture
model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer=Adam(learning_rate=1e-4))
print(model.summary())

early_stopping = EarlyStopping(patience=5, monitor='val_loss')
history = model.fit(X_train_scaled,
                    Y_train,
                    epochs=200,
                    validation_split=0.2,
                    callbacks=[early_stopping])
model.save('Brain_MRI_Aug.keras')

# Drawing graph of loss change on every epoch
plt.plot(history.history['loss'], label='Потеря обучения')
plt.plot(history.history['val_loss'], label='Потеря валидации')
plt.xlabel('Эпохи')
plt.ylabel('Функция потери')
plt.legend()
plt.show()

# Drawing graph of accuracy change on every epoch
plt.plot(history.history['accuracy'], label='Точность на обучающей выборке')
plt.plot(history.history['val_accuracy'], label='Точность на валидационной выборке')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()

