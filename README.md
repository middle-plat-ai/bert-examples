# bert-demo

BERT

bert：github上的bert的源代码

chinese_L-12_H-768_A-12：预训练的中文模型目录

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
