import MeCab

# MeCabオブジェクトの生成
tagger = MeCab.Tagger()

# 形態素解析
result = tagger.parse("メイが恋ダンスを踊っている。")
print(result)

