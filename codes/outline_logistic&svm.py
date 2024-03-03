import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix

def load_images_from_folder(folder, mask=False, target_size=(64, 64)):
    images = []
    if mask:
        for filename in os.listdir(folder):
            if (filename.endswith(".jpg") or filename.endswith(".png")) and "mask" in filename:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, target_size) / 255.0  # 归一化
                images.append(img)
    else:
        for filename in os.listdir(folder):
            if (filename.endswith(".jpg") or filename.endswith(".png")) and "mask" not in filename:
                img = cv2.imread(os.path.join(folder, filename), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, target_size) / 255.0  # 归一化
                images.append(img)
    return images

def extract_features(images):
    return [img.flatten() for img in images]

# 路径设置
base_path = "C:/Users/20134/Desktop/final pj/sta221/archive/Dataset_BUSI_with_GT"
categories = ['benign', 'malignant', 'normal']

# 遍历每个分类并生成标签
all_features = []
labels = []
for category in categories:
    folder_path = os.path.join(base_path, category)
    images = load_images_from_folder(folder_path,mask=True)
    features = extract_features(images)
    all_features.extend(features)
    labels.extend([category]* len(images))

# 转换为NumPy数组
features_matrix = np.array(all_features)
print(features_matrix.shape)

# 将标签转换为数字
label_dict = {'benign': 0, 'malignant': 1, 'normal': 2}
encoded_labels = [label_dict[label] for label in labels]

# 划分训练集和测试集 是一个2/8分
X_train, X_test, y_train, y_test = train_test_split(features_matrix, encoded_labels, test_size=0.2, random_state=42)

# 训练与调参（x.var=0.05)
svm_model = SVC()
param_grid = {'C': [5,10,20],
              'kernel': ['rbf', 'poly', 'sigmoid'],
              'gamma': [0.005,0.0025,0.0075,'scale']}

grid_search = GridSearchCV(svm_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

print("Best Parameters: ", grid_search.best_params_)

best_svm_model = grid_search.best_estimator_
best_svm_model.fit(X_train, y_train)
y_pred = best_svm_model.predict(X_test)

#结果
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)

class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)

logistic_model = LogisticRegression(max_iter=10000, solver='liblinear')
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': ['l1', 'l2']}

# 调参
grid_search = GridSearchCV(logistic_model, param_grid, cv=5)
grid_search.fit(X_train, y_train)

# 输出最佳参数
print("Best Parameters: ", grid_search.best_params_)

# 使用最佳参数的模型进行训练
best_logistic_model = grid_search.best_estimator_
best_logistic_model.fit(X_train, y_train)
y_pred = best_logistic_model.predict(X_test)

#结果
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", conf_matrix)
class_report = classification_report(y_test, y_pred)
print("Classification Report:\n", class_report)