export BERT_BASE_DIR=/home/itibia/chinese_L-12_H-768_A-12

python train_eval.py \
  --task_name cnews \
  --do_train  \
  --do_eval  \
  --data_dir ./data/ \
  --vocab_file $BERT_BASE_DIR/vocab.txt \
  --bert_config_file $BERT_BASE_DIR/bert_config.json \
  --init_checkpoint $BERT_BASE_DIR/bert_model.ckpt \
  --max_seq_length 150 \
  --train_batch_size 32 \
  --learning_rate 3e-5 \
  --num_train_epochs 5.0 \
  --output_dir ./output/ \
  --local_rank 3

