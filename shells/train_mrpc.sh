#!/bin/bash

export BERT_BASE_DIR=/data/NLP_projects/BERT/uncased_L-12_H-768_A-12
export GLUE_DIR=/data/NLP_projects/BERT/glue_data

cd /data/NLP_projects/BERT/bert

python run_classifier.py \
  --task_name=cola \
  --do_train=true \
  --do_eval=true \
  --data_dir=$GLUE_DIR/QQP \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length=80 \
  --train_batch_size=16 \
  --learning_rate=2e-3 \
  --num_train_epochs=10.0 \
  --output_dir=/data/NLP_projects/BERT/tmp/mrpc_output/
