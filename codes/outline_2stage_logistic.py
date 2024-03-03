import os
import cv2
import re
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_folder(folder, target_size=(64, 64)):
    images = []
    images_mask = []
    for filename in sorted(os.listdir(folder), key=lambda x: int(re.search(r'\((\d+)\)', x).group(1))):
        if (filename.endswith(".jpg") or filename.endswith(".png")) and "mask" in filename:
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size) / 255.0  # 归一化
            images_mask.append(img)
        elif (filename.endswith(".jpg") or filename.endswith(".png")) and "mask" not in filename:
            img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, target_size) / 255.0  # 归一化
            images.append(img)
    return images,images_mask

def extract_features(images):
    return [img.flatten() for img in images]

# 路径设置
base_path = "C:/Users/20134/Desktop/final pj/sta221/archive/Dataset_BUSI_with_GT_mask"
categories = ['benign', 'malignant', 'normal']

# 遍历每个分类并生成标签
all_features = []
all_features_mask = []
labels = []
for category in categories:
    folder_path = os.path.join(base_path, category)
    images,images_mask = load_images_from_folder(folder_path)
    features = extract_features(images)
    all_features.extend(features)
    features_mask = extract_features(images_mask)
    all_features_mask.extend(features_mask)
    labels.extend([category]* len(images))

# 转换为NumPy数组
features_matrix = np.array(all_features)
features_matrix_mask = np.array(all_features_mask)
print(features_matrix.shape)

#将标签转换为数字
label_dict = {'benign': 1, 'malignant': 2, 'normal': 3}
encoded_labels = [label_dict[label] for label in labels]
label_dict_mask = {'benign': 0, 'malignant': 0, 'normal': 3}
encoded_labels_mask = [label_dict_mask[label] for label in labels]

# 划分训练集和测试集 是一个2/8分
#use all mask data for classification between nomal and tumor
features_matrix_mask=np.array(features_matrix_mask)
encoded_labels_mask=np.array(encoded_labels_mask)
features_matrix=np.array(features_matrix)
encoded_labels=np.array(encoded_labels)
X_train_mask, X_test_mask, y_train_mask, y_test_mask = train_test_split(features_matrix_mask, encoded_labels_mask, test_size=0.2, random_state=42)

# use tumor original data to classify benign and malignant
X_train, X_test, y_train, y_test = train_test_split(features_matrix, encoded_labels, test_size=0.2, random_state=42)
fil = (y_train != 3 )
X_train_tumor = X_train[fil]
y_train_tumor = y_train[fil]

# first stage model
logistic_model1 = LogisticRegression(max_iter=10000, solver='liblinear')
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

# 调参
grid_search = GridSearchCV(logistic_model1, param_grid, cv=5)
grid_search.fit(X_train_mask, y_train_mask)

# 输出最佳参数
print("Best Parameters of first stage: ", grid_search.best_params_)

# 使用最佳参数的模型进行训练
best_logistic_model = grid_search.best_estimator_
best_logistic_model.fit(X_train_mask, y_train_mask)
y_pred1 = best_logistic_model.predict(X_test_mask)

# 结果
conf_matrix = confusion_matrix(y_test_mask, y_pred1)
print("Confusion Matrix of first stage:\n", conf_matrix)
class_report = classification_report(y_test_mask, y_pred1)
print("Classification Report of first stage:\n", class_report)
print("----------------------------------------------------------")

# second stage model
logistic_model2 = LogisticRegression(max_iter=10000, solver='liblinear')
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

# 调参
grid_search2 = GridSearchCV(logistic_model2, param_grid, cv=5)
grid_search2.fit(X_train_tumor, y_train_tumor)

# 输出最佳参数
print("Best Parameters of second stage: ", grid_search2.best_params_)

# 使用最佳参数的模型进行训练
best_logistic_model2 = grid_search2.best_estimator_
best_logistic_model2.fit(X_train_tumor, y_train_tumor)
y_pred2 = best_logistic_model2.predict(X_test)

fil2 = (y_pred1 == 3)
y_pred2[fil2] = 0
y_pred = y_pred1 + y_pred2

# stage 2
conf_matrix = confusion_matrix(y_test[~fil2], y_pred2[~fil2])
print("Confusion Matrix of second stage:\n", conf_matrix)
class_report = classification_report(y_test[~fil2], y_pred2[~fil2])
print("Classification Report of second stage:\n", class_report)
print("----------------------------------------------------------")

# together
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix of stage 1&2:\n", conf_matrix)
class_report = classification_report(y_test, y_pred)
print("Classification Report of stage 1&2:\n", class_report)
