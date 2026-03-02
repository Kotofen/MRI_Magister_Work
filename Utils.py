import os
from dotenv import load_dotenv
import cv2
import random
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix


# method for loading images from specified folder with specified labels
def load_images(label_map):
    # creating two lists to store the images and labels
    images = []
    labels = []
    load_dotenv()
    folder = os.getenv("DATASET_FOLDER")
    print(folder)

    # loading the images from each subfolder in folder of the dataset
    for subfolder in os.listdir(folder):
        category_path = os.path.join(folder, subfolder)
        if os.path.isdir(category_path):
             if subfolder in label_map:  # Check if the subfolder is present in the label_map
                label = label_map[subfolder]
                file_list = os.listdir(category_path)
                for filename in file_list:
                    img_path = os.path.join(category_path, filename)
                    image = cv2.imread(img_path)
                    # resizing the images to create a standard and so that it can be suitable for the model input
                    image = cv2.resize(image, (224, 224))
                    # cv2 reads the image as BGR so we need to convert it back to RGB
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    labels.append(label)

    return np.array(images), np.array(labels)


# method for loading images from specified folder with specified labels
def load_images_augmented(label_map):
    # creating two lists to store the images and labels
    images = []
    labels = []
    load_dotenv()
    folder = os.getenv("DATASET_FOLDER")
    print(folder)

    # loading the images from each subfolder in folder of the dataset
    for subfolder in os.listdir(folder):
        category_path = os.path.join(folder, subfolder)
        if os.path.isdir(category_path):
             if subfolder in label_map:  # Check if the subfolder is present in the label_map
                label = label_map[subfolder]
                file_list = os.listdir(category_path)
                for filename in file_list:
                    img_path = os.path.join(category_path, filename)
                    image = cv2.imread(img_path)
                    # Resizing the images to create a standard and so that it can be suitable for the model input
                    image = cv2.resize(image, (224, 224))
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    images.append(image)
                    labels.append(label)
                    # Rotating images
                    image_rotated_cw = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    images.append(image_rotated_cw)
                    labels.append(label)
                    image_rotated_ccw = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
                    images.append(image_rotated_ccw)
                    labels.append(label)
                    # Flipping images
                    image_flipped_x = cv2.flip(image, 0)
                    images.append(image_flipped_x)
                    labels.append(label)
                    image_flipped_y = cv2.flip(image, 1)
                    images.append(image_flipped_y)
                    labels.append(label)
                    # Changing contrast and brightness
                    image_converted1 = cv2.convertScaleAbs(image, alpha=1.5, beta=1)
                    images.append(image_converted1)
                    labels.append(label)
                    image_converted2 = cv2.convertScaleAbs(image, alpha=0.5, beta=1)
                    images.append(image_converted2)
                    labels.append(label)
                    image_converted3 = cv2.convertScaleAbs(image, alpha=1, beta=50)
                    images.append(image_converted3)
                    labels.append(label)
                    image_converted4 = cv2.convertScaleAbs(image, alpha=1, beta=-75)
                    images.append(image_converted4)
                    labels.append(label)

    return np.array(images), np.array(labels)


# method to show few images from dataset
def show_images(images, labels):
    plt.figure(figsize=(15, 5))
    # Display images with label 'yes'
    for i in range(3):
        plt.subplot(2, 3, i + 1)
        plt.imshow(images[labels == 1][random.randint(0, 50)])  # Filter images with label 'yes'
        plt.title("С отклонениями")
        plt.axis('off')
    # Display images with label 'no'
    for i in range(3):
        plt.subplot(2, 3, i + 4)
        plt.imshow(images[labels == 0][random.randint(0, 50)])  # Filter images with label 'no'
        plt.title("Без отклонений")
        plt.axis('off')
    plt.tight_layout()
    plt.show()


# method to show labels distribution
def show_labels_distribution(labels):
    unique_labels, label_counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(10, 5))
    plt.bar(unique_labels, label_counts, color=['blue', 'orange'])
    plt.xticks(unique_labels, ['Без отклонений', 'С отклонениями'])
    plt.xlabel('Метки')
    plt.ylabel('Кол-во')
    plt.title('Распределение изображений по меткам')
    plt.show()


# method to crop images
def crop_brain_region(image, size):
    # Converting the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # Applying Gaussian blur to smooth the image and reduce noise
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    # Thresholding the image to create a binary mask
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    # Performing morphological operations to remove noise
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    # Finding contours in the binary mask
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Assuming the brain part of the image has the largest contour
    c = max(contours, key=cv2.contourArea)
    # Getting the bounding rectangle of the brain part
    x, y, w, h = cv2.boundingRect(c)
    # Cropping the image around the bounding rectangle
    cropped_image = image[y:y + h, x:x + w]
    # Resizing cropped image to the needed size
    resized_image = cv2.resize(cropped_image, size)
    return resized_image


# method to show differences between original and cropped images
def show_cropped_differences(num, orig_images, cropped_images):
    plt.figure(figsize=(15, 5))
    for i in range(num):
        plt.subplot(2, num, i + 1)
        plt.imshow(orig_images[i])
        plt.title("Оригинальное")
        plt.axis("off")
    for i in range(num):
        plt.subplot(2, num, num + i + 1)
        plt.imshow(cropped_images[i])
        plt.title("Обрезанное")
        plt.axis("off")
    plt.show()


# method to show confusion matrix between predictions and test values
def show_confusion_matrix(y_test, binary_predictions, title):
    conf_matrix = confusion_matrix(y_test, binary_predictions)
    accuracy = accuracy_score(y_test, binary_predictions)
    print("Точность на тестовых данных: {:.3f} %".format(accuracy*100))
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Предсказанные метки')
    plt.ylabel('Настоящие метки')
    plt.show()


def copy_images_to_folder(images, labels, folder):
    label_map_decoded = {1: 'yes', 0: 'no'}
    for i, (image, label) in enumerate(zip(images, labels)):
        class_name = label_map_decoded[label]
        class_folder = os.path.join(folder, class_name)
        os.makedirs(class_folder, exist_ok=True)
        img_filename = f'{class_name}_{i}.jpg'  # Assuming images are in JPG format
        img_path = os.path.join(class_folder, img_filename)
        cv2.imwrite(img_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
