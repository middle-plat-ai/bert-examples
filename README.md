# bert-demo

[BERT](https://github.com/google-research/bert)


### 预训练模型下载：

[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip): 12-layer, 768-hidden, 12-heads, 110M parameters

[BERT-Large, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-24_H-1024_A-16.zip): 24-layer, 1024-hidden, 16-heads, 340M parameters

[BERT-Base, Cased](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip): 12-layer, 768-hidden, 12-heads , 110M parameters

[BERT-Large, Cased]: 24-layer, 1024-hidden, 16-heads, 340M parameters (Not available yet. Needs to be re-generated).

[BERT-Base, Multilingual](https://storage.googleapis.com/bert_models/2018_11_03/multilingual_L-12_H-768_A-12.zip): 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters

[BERT-Base, Chinese](https://storage.googleapis.com/bert_models/2018_11_03/chinese_L-12_H-768_A-12.zip): Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

### dir tree

```
bert-demo/
├── [glue_data](https://gluebenchmark.com/tasks)    ## 下载的官方glue数据
│   ├── MSRParaphraseCorpus  # rasa nlu train data
│   ├── zhTTQ                # atec-NLP数据
│   └── ...
├── bert                      # github上的bert的源代码
├── shells                    # 运行的脚本文件
├── tmp                       # 结果目录
├── requirement.txt           # run nlu and core server
└── README.md                 # readme file

```

bert：github上的bert的源代码

[glue_data](https://gluebenchmark.com/tasks)：下载的官方glue数据

shells：运行的脚本

tmp：运行结果目录

### mrpc

测试数据(英文)

glue_data/MSRParaphraseCorpus

训练直接运行：

sh shells/train_mrpc.sh

测试结果：

sh shells/predict_mrpc.sh

中文

glue_data/zhTTQ([atec-NLP数据](https://dc.cloud.alipay.com/index#/topic/intro?id=3))

训练直接运行：

sh shells/train_mrpc_zh.sh

测试结果：

sh shells/predict_mrpc_zh.sh

注：运行之前，需要先修改shell脚本里面的BERT_BASE_DIR、GLUE_DIR和TRAINED_CLASSIFIER路径。
