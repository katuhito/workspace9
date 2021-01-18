import matplotlib.pyplot as plt
import pandas as pd

# ワインデータの読み込み
wine = pd.read_csv("winequality-white.csv", sep=";", encoding="utf-8")

# 品質データごとにグループに分けて、その数をカウントする
count_data = wine.groupby('quality')["quality"].count()
print(count_data)

# カウントしたデータをグラフに描画
count_data.plot()
plt.savefig("wine-count-plt.png")
plt.show()
