import matplotlib.pyplot as plt
import cv2
from mosaic import mosaic as mosaic

# 画像に読み込んでモザイクを欠ける
img = cv2.imread("girl.jpg")
mos = mosaic(img, (50, 50, 450, 450), 10)

# モザイクをかけた画像を出力
cv2.imwrire("cat-mosaic.png", mos)
plt.imshow(cv2.cvtColor(mos, cv2.COLOR_BGR2RGB))
plt.show()
