import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Dense, Flatten, Dropout, MaxPooling2D, BatchNormalization, GlobalAveragePooling2D
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,LearningRateScheduler,ReduceLROnPlateau
import Utils


# Loading of initial dataset
label_map = {'no': 0, 'yes': 1}  # Used to map 0 for negatives and 1 for positives
images, labels = Utils.load_images(label_map)

print("Shape of images:", images.shape)  # Prints dataset info: (Num_Of_Imgs, img_height, img_width, [rgb_values])
print("Shape of labels:", labels.shape)  # Prints number of labels for imgs

Utils.show_images(images, labels)  # Shows few images from initial dataset
Utils.show_labels_distribution(labels)  # Shows label distribution of initial dataset

# Cropping MRI images for better brain visibility
all_cropped_images = []
for img in images:
    cropped_img = Utils.crop_brain_region(img, (224, 224))
    all_cropped_images.append(cropped_img)
all_cropped_images = np.array(all_cropped_images)

Utils.show_cropped_differences(5, images, all_cropped_images)  # Shows difference between initial imgs and cropped

# Splitting data on test and training sets
X_train, X_test, y_train, y_test = train_test_split(all_cropped_images, labels, test_size=0.2, shuffle=True, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42)

# Normalizing values of RGB in x from 0-255 to 0-1
X_train_scaled=X_train/255
X_test_scaled=X_test/255
X_val_scaled=X_val/255

# Adding layers to model
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

# Adding early stopping to training process
early_stopping = EarlyStopping(patience=5, monitor='val_loss')
# Training model
history = model.fit(X_train_scaled,
                    y_train,
                    batch_size=32,
                    epochs=50,
                    validation_data=(X_val_scaled,y_val),
                    callbacks=[early_stopping])
model.save('Brain_MRI.keras')

# Drawing training graph (loss on every epoch)
plt.plot(history.history['loss'], label='Потеря обучения')
plt.plot(history.history['val_loss'], label='Потеря валидации')
plt.xlabel('Эпохи')
plt.ylabel('Функция потери')
plt.legend()
plt.show()

# Drawing training graph (accuracy on every epoch
plt.plot(history.history['accuracy'], label='Точность на обучающей выборке')
plt.plot(history.history['val_accuracy'], label='Точность на валидационной выборке')
plt.xlabel('Эпохи')
plt.ylabel('Точность')
plt.legend()
plt.show()
