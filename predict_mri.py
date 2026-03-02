import cv2
import keras
import Utils
import numpy as np

img_path = r"C:\Users\ivka2\PycharmProjects\MRI\brain_tumor_dataset\yes\Y23.jpg"
labels = {0: "Без отлконений", 1: "С отклонениями"}
image = cv2.imread(img_path)
image = cv2.resize(image, (224, 224))
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cropped_image = Utils.crop_brain_region(image, (224, 224))
test_list = []
test_list.append(cropped_image)
test_list = np.array(test_list)

model = keras.src.saving.load_model('Brain_MRI_VGG16_Augmented.keras')
prediction = model.predict(test_list)
threshold = 0.5
binary_prediction = (prediction > threshold).astype(int)
print(labels)
print("Predicted category: ", labels[binary_prediction[0][0]])
