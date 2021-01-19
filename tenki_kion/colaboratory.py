# ダウンロード
from urllib.request import urlretrieve
urlretrieve("https://raw.githubusercontent.com/kujirahand/book-mlearn-gyomu/master/src/ch2/tenki/kion10y.csv", "kion10y.csv")

# データを表示
import pandas as pd
pd.read_csv("kion10y.csv")