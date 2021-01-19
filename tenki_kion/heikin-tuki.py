import matplotlib.pyplot as plt
import pandas as pd

# CSVを読み込む
df = pd.read_csv("kion10y.csv", encoding="utf-8")

# 月ごとに平均を求める
g = df.groupby(['月'])["気温"]
gg = g.sum() / g.count()

# 結果を出力
print(gg)
gg.plot()
plt.savefig("tenki-heikin-tuki.png")
plt.show()
