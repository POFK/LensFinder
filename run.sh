#!/bin/bash
#python main.py --lr 0.0001 --weight_decay 0.0 --name torch_003 --epochs 50
#python main.py --lr 0.0001 --weight_decay 0.0001 --name torch_004 --epochs 50
#python main.py --lr 0.0001 --weight_decay 0.001 --name torch_005 --epochs 50
#python main.py --lr 0.0001 --weight_decay 0.01 --name torch_006 --epochs 50
#python main.py --lr 0.0001 --weight_decay 0.1 --name torch_007 --epochs 50


#python main.py --lr 0.00001 --weight_decay 0.01 --name torch_008 --epochs 50 --test_batch_size 1024
#python main.py --lr 0.00001 --weight_decay 0.10 --name torch_009 --epochs 50 --test_batch_size 1024


python main.py --lr 0.00001 --weight_decay 0.01 --name lens_001 --epochs 50 --test_batch_size 1024

#eval:
python main.py  --name lens_001 --mode eval --epoch 20
