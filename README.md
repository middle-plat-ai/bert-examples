# bert-demo

BERT

预训练模型下载：

[BERT-Base, Uncased](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip): 12-layer, 768-hidden, 12-heads, 110M parameters

BERT-Large, Uncased: 24-layer, 1024-hidden, 16-heads, 340M parameters

BERT-Base, Cased: 12-layer, 768-hidden, 12-heads , 110M parameters

BERT-Large, Cased: 24-layer, 1024-hidden, 16-heads, 340M parameters (Not available yet. Needs to be re-generated).

BERT-Base, Multilingual: 102 languages, 12-layer, 768-hidden, 12-heads, 110M parameters

BERT-Base, Chinese: Chinese Simplified and Traditional, 12-layer, 768-hidden, 12-heads, 110M parameters

bert：github上的bert的源代码

glue_data：下载的官方glue数据

shells：运行的脚本

tmp：运行结果目录

mrpc

测试数据(英文)

glue_data/MSRParaphraseCorpus

训练直接运行：

sh shells/train_mrpc.sh

测试结果：

sh shells/predict_mrpc.sh

中文

glue_data/zhTTQ

训练直接运行：

sh shells/train_mrpc_zh.sh

测试结果：

sh shells/predict_mrpc_zh.sh

注：运行之前，需要先修改shell脚本里面的BERT_BASE_DIR、GLUE_DIR和TRAINED_CLASSIFIER路径。

[Spring-data-jpa 查询  复杂查询陆续完善中](http://www.cnblogs.com/sxdcgaq8080/p/7894828.html)
