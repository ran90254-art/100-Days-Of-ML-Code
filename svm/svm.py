```python
# ============================================
# SVM Classification - Social Network Ads
# ============================================

# 1. 导入需要的库
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score


# ============================================
# 2. 读取数据
# ============================================

dataset = pd.read_csv("Social_Network_Ads.csv")

print("数据前 5 行：")
print(dataset.head())

print("\n数据形状：")
print(dataset.shape)

print("\n列名：")
print(dataset.columns)


# ============================================
# 3. 选择特征 X 和目标 y
# ============================================

# X：模型的输入
# 使用 Age 和 EstimatedSalary 两个特征
X = dataset[["Age", "EstimatedSalary"]].values

# y：模型需要预测的结果
# Purchased = 0：没有购买
# Purchased = 1：购买
y = dataset["Purchased"].values


print("\nX 的形状：")
print(X.shape)

print("\ny 的形状：")
print(y.shape)


# ============================================
# 4. 划分训练集和测试集
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=0
)

print("\n训练集大小：")
print(X_train.shape)

print("\n测试集大小：")
print(X_test.shape)


# ============================================
# 5. Feature Scaling（特征标准化）
# ============================================

sc = StandardScaler()

# 对训练集：
# fit：学习训练集的均值和标准差
# transform：使用均值和标准差进行标准化
X_train = sc.fit_transform(X_train)

# 对测试集：
# 只能 transform，不能重新 fit
X_test = sc.transform(X_test)


# ============================================
# 6. 创建 SVM 分类器
# ============================================

classifier = SVC(
    kernel="linear",
    random_state=0
)


# ============================================
# 7. 使用训练集训练 SVM
# ============================================

classifier.fit(X_train, y_train)


# ============================================
# 8. 使用测试集进行预测
# ============================================

y_pred = classifier.predict(X_test)


print("\n真实结果：")
print(y_test)

print("\n预测结果：")
print(y_pred)


# ============================================
# 9. 计算混淆矩阵
# ============================================

cm = confusion_matrix(y_test, y_pred)

print("\n混淆矩阵：")
print(cm)


# ============================================
# 10. 计算准确率
# ============================================

accuracy = accuracy_score(y_test, y_pred)

print("\n模型准确率：")
print(f"{accuracy:.2%}")


# ============================================
# 11. 可视化测试集的分类结果
# ============================================

# 创建网格
import numpy as np

X_set, y_set = X_test, y_test

X1, X2 = np.meshgrid(
    np.arange(
        start=X_set[:, 0].min() - 1,
        stop=X_set[:, 0].max() + 1,
        step=0.01
    ),
    np.arange(
        start=X_set[:, 1].min() - 1,
        stop=X_set[:, 1].max() + 1,
        step=0.01
    )
)


# 预测网格中每一个点属于哪一类
Z = classifier.predict(
    np.array([X1.ravel(), X2.ravel()]).T
)

Z = Z.reshape(X1.shape)


# 绘制决策区域
plt.contourf(
    X1,
    X2,
    Z,
    alpha=0.3
)


# 绘制测试数据点
for i, j in enumerate(np.unique(y_set)):

    plt.scatter(
        X_set[y_set == j, 0],
        X_set[y_set == j, 1],
        label=j
    )


plt.title("SVM Classification - Test Set")
plt.xlabel("Age (Standardized)")
plt.ylabel("Estimated Salary (Standardized)")
plt.legend()

plt.show()
```
