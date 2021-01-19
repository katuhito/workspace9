in_file = "data.csv"
out_file = "kion10y.csv"

# CSVファイルを1行ずつ読み込み
with open(in_file, "rt", encoding="Shift_JIS") as fr:
    lines = fr.readlines()

# ヘッダーをそぎ落として、新たなヘッダーを付ける
lines = ["年, 月, 日, 気温, 品質, 均質\n"] + lines[5:]
lines = map(lambda v: v.replace('/', ','), lines)
result = "".join(lines).strip()
print(result)

# 結果をファイルへ出力
with open(out_file, "wt", encoding="utf-8") as fw:
    fw.write(result)
    print("saved.")
