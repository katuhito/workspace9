import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
from sklearn.utils.testing import all_estimators
from sklearn.model_selection import KFold
import warnings
from sklearn.model_selection import cross_val_score

# アヤメデータの読み込み
iris_data = pd.read_csv("iris.csv", encoding="utf-8")

# アヤメデータをラベルと入力データに分離する
y = iris_data.loc[:, "Name"]
x = iris_data.loc[:, ["SepalLength", "SepalWidth", "PetalLength", "PetalWidth"]]

# 学習用とテスト用に分離する
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, train_size = 0.8, shuffle = True)

# classifierのアルゴリズムをすべて取得する
allAlgorithms = all_estimators(type_filter="classifier")
# warnings.simplefilter("error")

# K分割クロスバリデーション用オブジェクト
kfold_cv = KFold(n_splits=5, shuffle=True)
warnings.filterwarnings('ignore')

for(name, algorithm) in allAlgorithms:
    try:
        # 各アルゴリズムのオブジェクトを作成
        if(name == "LinearSVC"):
            clf = algorithm(max_iter = 10000)
        else:
            clf = algorithm()

        # scoreメソッドを持つクラスを対象とする
        if hasattr(clf, "score"):
            # クロスバリデーションを行う
            scores = cross_val_score(clf, x, y, cv=kfold_cv)
            print(name, "の正解率=")
            print(scores)

    # Exceptionは無視する
    except Exception as e :
        pass
    

        # 学習して評価する
        # clf.fit(x_train, y_train)
        # y_pred = clf.predict(x_test)
        # print(name, "の正解率 = " , accuracy_score(y_test, y_pred))
    
    # WarningやExceptionの内容を表示する
    # except Warning as w :
    #     print("\033[33m"+"Warning: "+"\033[0m", name, ":", w.args)
    # except Exception as e :
    #     pass
