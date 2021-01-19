import pandas as pd

# PandasでCSVを読み込む
df = pd.read_csv("kion10y.csv", encoding="utf-8")

# 日付ごとに気温をリストにまとめる
md = {}
for i, row in df.iterrows():
    m, d, v = (int(row['月']), int(row['日']), float(row['気温']))
    key = "{:02d}/{:02d}".format(m, d)
    if not(key in md): md[key] = []
    md[key] += [v]

avs = {}
for key in sorted(md):
    v = avs[key] = sum(md[key]) / len(md[key])
    print("{0} : {1}".format(key, v))
     