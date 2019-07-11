#!/bin/bash
#python lens/main.py --lr 0.001 --name lens_011 --epochs 30 --weight_decay 0.0 
#python lens/main.py --lr 0.00001 --name lens_012 --epochs 30 --weight_decay 0.0 
#python lens/main.py --lr 0.0001 --name lens_013 --epochs 30 --weight_decay 0.0 
#python lens/main.py --lr 0.0001 --name lens_014 --epochs 30 --weight_decay 0.0001 
#python lens/main.py --lr 0.0001 --name lens_015 --epochs 30 --weight_decay 0.001 
#python lens/main.py --lr 0.0001 --name lens_016 --epochs 30 --weight_decay 0.01 
#python lens/main.py --lr 0.0001 --name lens_017 --epochs 30 --weight_decay 0.1 
#python main.py --lr 0.00001 --weight_decay 0.01 --name torch_008 --epochs 50 --test_batch_size 1024
#python main.py --lr 0.00001 --weight_decay 0.10 --name torch_009 --epochs 50 --test_batch_size 1024
#--------------------- run @ 0710 -----------------------------------------------
#python lens/main.py --lr 0.00001 --weight_decay 0.1 --name lens_018 --epochs 30 --test_batch_size 1024
#python lens/main.py --lr 0.00001 --weight_decay 1.0 --name lens_019 --epochs 30 --test_batch_size 1024
#python lens/main.py --lr 0.00001 --weight_decay 0.01 --name lens_020 --epochs 30 --test_batch_size 1024
#python lens/main.py --lr 0.00001 --weight_decay 10.0 --name lens_021 --epochs 100 --test_batch_size 1024
#python lens/main.py --lr 0.00001 --weight_decay 0.1 --name lens_022 --epochs 100 --test_batch_size 1024
#python lens/main.py --lr 0.00001 --weight_decay 1.0 --name lens_023 --epochs 100 --test_batch_size 1024
#python lens/main.py --lr 0.000003 --weight_decay 1.0 --name lens_023 --epochs 100 --epoch 25 --test_batch_size 1024 --batch_size=256
#python lens/main.py --lr 0.0001 --weight_decay 0.1 --name lens_026 --epochs 30 --test_batch_size 1024 --batch_size=64
#python lens/main.py --lr 0.00001 --weight_decay 0.01 --name lens_028 --epochs 50 --test_batch_size 1024 --batch_size=1024 --epoch 46
#python lens/main.py --lr 0.00001 --weight_decay 0.1 --name lens_030 --epochs 100 --test_batch_size 2048  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.1 --name lens_031 --epochs 50 --test_batch_size 512  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 3.0 --name lens_032 --epochs 50 --test_batch_size 512  --batch_size 128
#--------------------- run @ 0711 -----------------------------------------------
#python lens/main.py --lr 0.00001 --weight_decay 0.5 --name lens_033 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.2 --name lens_034 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.1 --name lens_035 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.1 --name lens_036 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.1 --name lens_037 --epochs 50 --test_batch_size 1024  --batch_size 64
#python lens/main.py --lr 0.00001 --weight_decay 0.1 --name lens_038 --epochs 50 --test_batch_size 1024  --batch_size 256
#python lens/main.py --lr 0.00001 --weight_decay 0.05 --name lens_039 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.000001 --weight_decay 0.5 --name lens_040 --epochs 50 --test_batch_size 1024  --batch_size 256 --epoch 50
#python lens/main.py --lr 0.00001 --weight_decay 0.5 --name lens_041 --epochs 30 --test_batch_size 1024  --batch_size 1024 --epoch 60
#python lens/main.py --lr 0.00001 --weight_decay 0.5 --name lens_042 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.5 --name lens_043 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.5 --name lens_044 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.5 --name lens_045 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.5 --name lens_046 --epochs 50 --test_batch_size 1024  --batch_size 128
#python lens/main.py --lr 0.00001 --weight_decay 0.5 --name lens_047 --epochs 50 --test_batch_size 1024  --batch_size 256
#python lens/main.py --lr 0.000001 --weight_decay 0.5 --name lens_047 --epochs 50 --test_batch_size 1024  --batch_size 256 --epoch 50



#--------------------- run @ 0710 -----------------------------------------------
#python lens/main.py --lr 0.00001 --weight_decay 0.01 --name lens_020 --epochs 30 --test_batch_size 1024
#eval:
#python main.py  --name lens_010 --mode eval --epoch 20
#--------------------- run @ 0711 -----------------------------------------------
python lens/main.py --lr 0.000001 --weight_decay 0.5 --name lens_047 --batch_size 256 --epoch 80 --mode eval
