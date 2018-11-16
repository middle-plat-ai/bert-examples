# bert-demo

[BERT](https://github.com/google-research/bert)


## 预训练模型下载：

[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip): 12-layer, 768-hidden, 12-heads, 110M parameters

[BERT-Large, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip): 24-layer, 1024-hidden, 16-heads, 340M parameters

[BERT-Base, Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip): 12-layer, 768-hidden, 12-heads , 110M parameters

[BERT-Large, Cased]: 24-layer, 1024-hidden, 16-heads, 340M parameters (Not available yet. Needs to be re-generated).

[BERT-Base, Multilingual](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip): 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters

[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip): Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

## dir tree

```
bert-demo/
├── glue_data                # 下载的官方glue数据
│   ├── MSRParaphraseCorpus  # rasa nlu train data
│   ├── zhTTQ                # atec-NLP数据
│   └── ...
├── bert                      # github上的bert的源代码
├── shells                    # 运行的脚本文件
├── tmp                       # 结果目录
├── requirement.txt           # run nlu and core server
└── README.md                 # readme file

```

[下载glue_data](https://gluebenchmark.com/tasks)


## Microsoft Research Paraphrase Corpus (MRPC) 

### 测试数据(英文)

[glue_data/MSRParaphraseCorpus](https://www.microsoft.com/en-us/download/details.aspx?id=52398&from=http%3A%2F%2Fresearch.microsoft.com%2Fen-us%2Fdownloads%2F607d14d9-20cd-47e3-85bc-a2f65cd28042%2F)

训练直接运行：

sh shells/train_mrpc.sh

测试结果：

sh shells/predict_mrpc.sh

### 中文

glue_data/zhTTQ([atec-NLP数据](https://dc.cloud.alipay.com/index#/topic/intro?id=3))

数据描述：

第一列是标签，第2列和第3列分别是string1和string2的编号，第4列和第5列分别是string1和实体ring2的###分词结果

Quality	#1 ID	#2 ID	#1 String	#2 String
1	35489	110883	晚上 *** 点 从 借呗 借钱 ， 多久 到账	借呗 要 多久 到账 的
0	39136	7574	蚂蚁 花呗 可以 推迟 几天 还 么	花呗 还款 十 日 之前 还是 可以 十 日 当天
0	33871	58091	花呗 的 学历 认证 ， 文化 低 的 怎么办	花呗 认证 怎么 该

训练直接运行：

sh shells/train_mrpc_zh.sh

测试结果：

sh shells/predict_mrpc_zh.sh

注：运行之前，需要先修改shell脚本里面的BERT_BASE_DIR、GLUE_DIR和TRAINED_CLASSIFIER路径。
