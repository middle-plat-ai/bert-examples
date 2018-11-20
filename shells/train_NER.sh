#!/bin/bash

export BERT_BASE_DIR=/data/NLP_projects/BERT/uncased_L-12_H-768_A-12
export GLUE_DIR=/data/NLP_projects/BERT/glue_data

cd ..

python BERT_NER.py   \
  --task_name="NER"  \
  --do_train=true   \
  --do_eval=True   \
  --data_dir=$GLUE_DIR/NERdata   \
  --vocab_file=$BERT_BASE_DIR/vocab.txt  \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   \
  --max_seq_length=64   \
  --train_batch_size=16   \
  --learning_rate=2e-5   \
  --num_train_epochs=3.0   \
  --output_dir=/data/NLP_projects/BERT/tmp/NER_output/
