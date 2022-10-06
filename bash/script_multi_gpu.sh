DATASET=trivia

CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=24 python -m torch.distributed.launch --nproc_per_node=2 ../train_reader.py \
--train_data ../data/json_input/${DATASET}_train.jsonl \
--eval_data ../data/json_input/${DATASET}_dev.jsonl \
--train_graph ../data/graphs/${DATASET}_train \
--eval_graph  ../data/graphs/${DATASET}_dev \
--ground_truth_for_test  ../data/json_input/${DATASET}_test_relation.json \
--relation_base ../data/json_input/${DATASET}_relations.json \
--test_data  ../data/json_input/${DATASET}_test.jsonl \
--test_graph ../data/graphs/${DATASET}_test \
--dataset ${DATASET} \
--model_size base \
--checkpoint_dir checkpoint \
--use_checkpoint \
--lr 0.00003 \
--optim adamw \
--scheduler linear \
--weight_decay 0.01 \
--text_maxlength 256 \
--per_gpu_batch_size 2 \
--n_context 100 \
--total_step 30000 \
--warmup_step 2000 \
--answer_maxlength 10 \
--gnn_dimension 1024 \
--is_distributed \
--name ${DATASET} \
--seed 2345 \
--gnn_layer 2 \
--gnn_dropout 0.2 \
--n_heads 8 \
--layer2insert 3 \
--eval_freq 500 \
--bpe 