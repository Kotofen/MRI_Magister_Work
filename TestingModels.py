import os
import cv2
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import Utils


# Models test on initial dataset images
# data_folder = r"brain_mri_dataset"
label_map = {'no': 0, 'yes': 1}
images, labels = Utils.load_images(label_map)
cropped_images = []
for img in images:
    cropped_img = Utils.crop_brain_region(img, (224, 224))
    cropped_images.append(cropped_img)
cropped_images = np.array(cropped_images)


_, X_test, _, Y_test = train_test_split(cropped_images, labels, test_size=0.2, shuffle=True)
X_test_scaled = X_test/255

model = keras.src.saving.load_model('Brain_MRI.keras')
predictions = model.predict(X_test_scaled)
threshold = 0.5
binary_predictions_m1 = (predictions > threshold).astype(int)
Utils.show_confusion_matrix(Y_test, binary_predictions_m1, 'Матрица ошибок базовой модели')

model2 = keras.src.saving.load_model('Brain_MRI_Aug.keras')
predictions = model2.predict(X_test_scaled)
threshold = 0.5
binary_predictions_m2 = (predictions > threshold).astype(int)
Utils.show_confusion_matrix(Y_test, binary_predictions_m2, 'Матрица ошибок улучшенной модели')

model3 = keras.src.saving.load_model('Brain_MRI_VGG16_Augmented.keras')
predictions = model3.predict(X_test_scaled)
threshold = 0.5
binary_predictions_m3 = (predictions > threshold).astype(int)
Utils.show_confusion_matrix(Y_test, binary_predictions_m3, 'Матрица ошибок модели на основе VGG16')

# Models test on augmented dataset images
augmented_images, augmented_img_labels = Utils.load_images_augmented(data_folder, label_map)
cropped_augmented_images = []
for img in augmented_images:
    cropped_img = Utils.crop_brain_region(img, (224, 224))
    cropped_augmented_images.append(cropped_img)
cropped_rotated_images = np.array(cropped_augmented_images)


_, X_test, _, Y_test = train_test_split(cropped_rotated_images, augmented_img_labels, test_size=0.2, shuffle=True)
X_test_scaled = X_test/255

predictions = model.predict(X_test_scaled)
threshold = 0.5
binary_predictions_m1 = (predictions > threshold).astype(int)
Utils.show_confusion_matrix(Y_test, binary_predictions_m1, 'Матрица ошибок базовой модели')

predictions = model2.predict(X_test_scaled)
threshold = 0.5
binary_predictions_m2 = (predictions > threshold).astype(int)
Utils.show_confusion_matrix(Y_test, binary_predictions_m2, 'Матрица ошибок улучшенной модели')

predictions = model3.predict(X_test_scaled)
threshold = 0.5
binary_predictions_m3 = (predictions > threshold).astype(int)
Utils.show_confusion_matrix(Y_test, binary_predictions_m3, 'Матрица ошибок модели на основе VGG16')