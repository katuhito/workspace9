import matplotlib.pyplot as plt
import pandas as pd

# ファイルを読む
df = pd.read_csv('kion10y.csv', encoding="utf-8")

# 気温が30度超えのデータを調べる
atui_bool = (df["気温"] > 30)

# データを抜き出す
atui = df[atui_bool]

# 年ごとにカウント
cnt = atui.groupby(["年"])["年"].count()

# 出力
print(cnt)
cnt.plot()
plt.savefig("tennki-over30.png")
plt.show()
