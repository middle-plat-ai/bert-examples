# bert-demo

[BERT](https://github.com/google-research/bert)：google-BERT


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

 训练

```
  sh shells/train_mrpc.sh
```

测试

```
  sh shells/predict_mrpc.sh
```

### 中文

glue_data/zhTTQ([atec-NLP数据](https://dc.cloud.alipay.com/index#/topic/intro?id=3))

数据描述：

```
Quality	#1 ID	#2 ID	#1 String	#2 String

0	39136	7574	蚂蚁 花呗 可以 推迟 几天 还 么	花呗 还款 十 日 之前 还是 可以 十 日 当天

第一列是标签，第2列和第3列分别是string1和string2的编号，第4列和第5列分别是string1和实体ring2的分词结果，分隔符为\t
```

训练

```
sh shells/train_mrpc_zh.sh
```

测试

```
sh shells/predict_mrpc_zh.sh
```

注：运行之前，需要先修改shell脚本里面的BERT_BASE_DIR、GLUE_DIR和TRAINED_CLASSIFIER路径。

## 命名实体识别(NER)

### 英文

数据：glue_data/NERdata

训练

```
python bert_ner.py
```

测试

```
python ner_predict.py
```

命令行测试

```
python predict_cli.py
```

## [bert-as-service](https://github.com/hanxiao/bert-as-service)

依赖
```
python>=3.5，不支持python2
tensorflow>=1.11(运行bert的需要)
```

安装
```
pip install bert-serving-server  # server
pip install bert-serving-client  # client, independent of `bert-serving-server`
```

启动服务
```
bert-serving-start -model_dir /tmp/english_L-12_H-768_A-12/ -num_worker=4
```

使用方法
```
from bert_serving.client import BertClient
bc = BertClient(ip='xx.xx.xx.xx')  # ip address of the GPU machine，如果是本机，可以不填
bc.encode(['First do it', 'then do it right', 'then do it better'])
```


## Reference

1 [BERT](https://github.com/google-research/bert)：google-BERT
 
2 [BERT-NER](https://github.com/kyzhouhzau/BERT-NER)

3 [BERT-NER-CLI](https://github.com/JamesGu14/BERT-NER-CLI)
 
4 [bert-as-service](https://github.com/hanxiao/bert-as-service)
