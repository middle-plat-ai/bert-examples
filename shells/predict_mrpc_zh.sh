#!/bin/bash

export BERT_BASE_DIR=/data/NLP_projects/BERT/uncased_L-12_H-768_A-12
export GLUE_DIR=/data/NLP_projects/BERT/glue_data
export TRAINED_CLASSIFIER=/data/NLP_projects/BERT/tmp/mrpc_output/

cd /data/NLP_projects/BERT/bert

python run_classifier.py \
  --task_name=MRPC \
  --do_predict=true \
  --data_dir=$GLUE_DIR/MSRParaphraseCorpus \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$TRAINED_CLASSIFIER \
  --max_seq_length=80 \
  --output_dir=/data/NLP_projects/BERT/tmp/mrpc_output/
