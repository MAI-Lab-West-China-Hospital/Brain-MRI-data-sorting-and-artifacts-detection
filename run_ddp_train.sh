#!/bin/bash

python -W ignore -m torch.distributed.launch --nproc_per_node=8 --nnodes=1 --node_rank=0 train.py --train_path=/PATH/TO/TRAIN/DATA --test_path=/PATH/TO/TEST/PATH --label_file=/POINT/TO/LABEL/FILE
