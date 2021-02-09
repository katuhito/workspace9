import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

# Unicodeのコードポイント頻度測定
def count_codePoint(str):
    # Unicodeのコードポイントをアドレスとする配列を用意
    counter = np.zeros(65535)

    for i in range(len(str)):
        # 各文字をUnicodeのコードポイントに変換
        code_point = ord(str[i])
        if code_point > 65535:
            continue
        # 対応するアドレスの出現回数をインクリメント
        counter[code_point] += 1

    # 各要素を文字数で割って正規化
    counter = counter/len(str)
    return counter

# 学習用データの準備
ja_str = 'これは日本語の文章です。'
en_str = 'This is English Sentences.'
th_str = 'นี่คือประโยคภาษาญี่ปุ่น' #タイ語

x_train = [count_codePoint(ja_str), count_codePoint(en_str), count_codePoint(th_str)]
y_train = ['ja', 'en', 'th']

# 学習する
clf = GaussianNB()
clf.fit(x_train, y_train)

# 評価用のデータの準備
ja_test_str = 'こんにちは'
en_test_str = 'Hello'
th_test_str = 'สวัสดี'

x_test = [count_codePoint(en_test_str), count_codePoint(th_test_str), count_codePoint(ja_test_str)]
y_test = ['en', 'th', 'ja']

# 評価する
y_pred = clf.predict(x_test)
print(y_pred)
print("正解率 = " , accuracy_score(y_test, y_pred))