import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# アヤメデータの読み込み
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# アヤメデータをラベルとデータに分離する
y_labels = iris_data.loc[:, "Name"]
x_data = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# ラベルデータをOne-Hotベクトルに直す
labels = {
    'Iris-setosa': [1, 0, 0],
    'Iris-versicolor': [0, 1, 0],
    'Iris-virginica': [0, 0, 1]
}
y_nums = np.array(list(map(lambda v : labels[v] , y_labels)))
x_data = np.array(x_data)

# 学習用とテスト用に分離する
x_train, x_test, y_train, y_test = train_test_split(x_data, y_nums, train_size=0.8)

# モデル構造を定義
Dense = keras.layers.Dense
model = keras.models.Sequential()
model.add(Dense(10, activation='relu' , input_shape=(4,)))
model.add(Dense(3, activation='softmax'))

# モデルを構築
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# アヤメの分類問題を修正
import matplotlib.pyplot as plt
epochs = 10

# 学習を実行
result = model.fit(x_train, y_train, batch_size=20, epochs=epochs)

# モデルを評価
score = model.evaluate(x_test, y_test, verbose=1)
print('正解率=', score[1], 'loss=', score[0])

# 学習の様子をグラフに描画
plt.plot(range(1, epochs+1), result.history['accuracy'], label="training")
plt.plot(range(1, epochs+1), result.history['loss'], label="loss")
plt.xlabel('Epochs=' + str(epochs))
plt.ylabel('Accuracy')
plt.legend()
plt.show()


# # 学習を実行
# model.fit(x_train, y_train, batch_size=20, epochs=300)

# # モデルを評価
# score = model.evaluate(x_test, y_test, verbose=1)
# print('正解率=', score[1], 'loss=', score[0])





