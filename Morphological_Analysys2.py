import MeCab

# mecab-ipadic-NEologed辞書を指定して、MeCabオブジェクトを生成
tagger = MeCab.Tagger('-d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd')

# 形態素解析
result = tagger.parse("メイが恋ダンスを踊っている。")
print(result)
