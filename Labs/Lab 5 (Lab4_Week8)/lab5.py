""" 1. Import any libraries you want"""

import tensorflow as tf
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, classification_report

# Build a histogram for each image
def build_histogram(descriptors, kmeans, k):
    histogram = np.zeros(k)
    if descriptors is not None:
        words = kmeans.predict(descriptors)
        for word in words:
            histogram[word] += 1
    return histogram

def main():
    """ 2. Download the dataset """
    train_ds = tf.keras.utils.image_dataset_from_directory("train",     image_size=(224, 224), batch_size=None, shuffle=False)
    val_ds   = tf.keras.utils.image_dataset_from_directory("validation",  image_size=(224, 224), batch_size=None, shuffle=False)
    test_ds  = tf.keras.utils.image_dataset_from_directory("test",      image_size=(224, 224), batch_size=None, shuffle=False)

    # Create SIFT model
    sift = cv2.SIFT_create()

    """ 3. Extract local features from the dataset """
    all_descriptors = []

    for image, label in train_ds:
        # Convert tensor to numpy array 
        img_np = image.numpy().astype(np.uint8)
        # Convert to grayscale for SIFT
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        # Extract descriptors of SIFT
        _, descriptors = sift.detectAndCompute(gray, None)
        if descriptors is not None:
            all_descriptors.append(descriptors)
    # Save all descriptors into a single array
    all_descriptors = np.vstack(all_descriptors)

    """ 4. Build your visual vocabulary """
    # Vocab cluster size
    k = 200
    kmeans = KMeans(n_clusters=k, random_state=42)
    # Cluster descriptors 
    kmeans.fit(all_descriptors)
    
    # Build per-image histograms for data sets
    X_train = []
    y_train = []

    # Collect histograms grouped by class
    class_names_train = train_ds.class_names
    class_histograms_train = defaultdict(lambda: np.zeros(k))
    class_counts_train = defaultdict(int)

    for image, label in train_ds:
        # Compute descriptors
        img_np = image.numpy().astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)

        # Build histogram using descriptors
        hist = build_histogram(descriptors, kmeans, k)
        X_train.append(hist)
        y_train.append(label.numpy())
        class_name = class_names_train[label.numpy()]

        # Save histogram and update class counts
        class_histograms_train[class_name] += hist
        class_counts_train[class_name] += 1
        
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    X_test = []
    y_test = []
    
    # Collect histograms grouped by class
    class_names_test = test_ds.class_names
    class_histograms_test = defaultdict(lambda: np.zeros(k))
    class_counts_test = defaultdict(int)

    for image, label in test_ds:
        # Compute descriptors
        img_np = image.numpy().astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)

        # Build histogram using descriptors
        hist = build_histogram(descriptors, kmeans, k)
        X_test.append(hist)
        y_test.append(label.numpy())
        class_name = class_names_test[label.numpy()]

        # Save histogram and update class counts
        class_histograms_test[class_name] += hist
        class_counts_test[class_name] += 1
        
    X_test = np.array(X_test)
    y_test = np.array(y_test)
    
    X_val = []
    y_val = []
    
    # Collect histograms grouped by class
    class_names_val = val_ds.class_names
    class_histograms_val = defaultdict(lambda: np.zeros(k))
    class_counts_val = defaultdict(int)

    for image, label in val_ds:
        # Compute descriptors
        img_np = image.numpy().astype(np.uint8)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        _, descriptors = sift.detectAndCompute(gray, None)

        # Build histogram using descriptors
        hist = build_histogram(descriptors, kmeans, k)
        X_val.append(hist)
        y_val.append(label.numpy())
        class_name = class_names_val[label.numpy()]

        # Save histogram and update class counts
        class_histograms_val[class_name] += hist
        class_counts_val[class_name] += 1
        
    X_val = np.array(X_val)
    y_val = np.array(y_val)

    """ 5. Plot average histogram per class """
    # One histogram per class
    _, axes = plt.subplots(1, len(class_names_train), figsize=(15, 4))
    for ax, class_name in zip(axes, class_names_train):
        # calculate per-class avergae histogram
        avg_histogram = class_histograms_train[class_name] / class_counts_train[class_name]
        ax.bar(range(k), avg_histogram)
        ax.set_title(class_name)
        ax.set_xlabel("Visual Word")
        ax.set_ylabel("Frequency")

    plt.suptitle("Average Visual Word Histogram per Class")
    plt.tight_layout()
    plt.show()
    
    """ 6. Select any classifier model and train them on the dataset"""
    # Train the logistic regression classifier
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    
    pred = lr.predict(X_test)
    # pred_val = lr.predict(X_val)
    
    print(classification_report(y_test, pred))
    # print(classification_report(y_val, pred_val))
    
    """ 7. Plot the performance metrics of your classifier model in terms of a confusion matrix """
    cm = confusion_matrix(y_test, pred)
    # cm_val = confusion_matrix(y_val, pred_val)
    
    sns.heatmap(cm, 
                annot=True,
                fmt='g', 
                xticklabels=['Cat', 'Cow', 'Deer', 'Dog', 'Lion'],
                yticklabels=['Cat', 'Cow', 'Deer', 'Dog', 'Lion'])
    plt.ylabel('Actual', fontsize=13)
    plt.title('Confusion Matrix', fontsize=17, pad=20)
    plt.gca().xaxis.set_label_position('top') 
    plt.xlabel('Prediction', fontsize=13)
    plt.gca().xaxis.tick_top()

    plt.gca().figure.subplots_adjust(bottom=0.2)
    plt.gca().figure.text(0.5, 0.05, 'Prediction', ha='center', fontsize=13)
    plt.show()

if __name__ == "__main__":
    main()