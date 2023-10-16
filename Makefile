train_resnet: 
	python lab2.py --optimizer sgd --epochs 5 --train_batch_size 128 --train_num_workers 2 --test_batch_size 100 --test_num_workers 2 --lr 0.1 --weight_decay 5e-4 --momentum 0.9 --verbose True --data_download_path ./data --cuda 1 --nesterov 0 --enable_torch_profiling 1 --include_batch_norm_layers 1

launch-tb:
	tensorboard --logdir=./log

install-deps:
	pip install torch_tb_profiler