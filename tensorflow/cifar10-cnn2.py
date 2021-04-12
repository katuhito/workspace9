import cv2
import numpy as numpy

labels = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
im_size = 32 * 32 * 3

# モデルデータを読み込み
model.load_weights('cifar10-mlp-weight.h5')

# OpenCVを使って画像を読み込む
im = cv2.imread('test-car.jpg')

# 色空間を変換してリサイズ
im = cv2.cvtcolor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (32, 32))
plt.imshow(im) #画像を出力
plt.show()

# MLPで学習した画像データに合わせる
im = im.reshape(im_size).astype('float32') / 255

# 予測する
r = model.predict(np,array([im]), batch_size=32, verbose=1)
res = r[0]

# 結果を表示する
for i, acc in enumerate(res):
    print(labels[i], "=", int(acc * 100))
print("---")
print("予測した結果=", labels[res.argmax()])
