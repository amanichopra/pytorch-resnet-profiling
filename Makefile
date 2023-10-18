OPTIMIZER = "sgd"
EPOCHS = 5
TRAIN_BATCH_SIZE = 128
TRAIN_NUM_WORKERS = 2
TEST_BATCH_SIZE = 100
TEST_NUM_WORKERS = 2
LR = 0.1
WEIGHT_DECAY = "5e-4"
MOMENTUM = 0.9
DATA_DOWNLOAD_PATH = "./data"
CUDA = 1

OPT_NUM_WORKERS = 20

install-deps:
	pip install ipykernel notebook torch_tb_profiler

c1: lab2.py
	python lab2.py --optimizer $(OPTIMIZER) --epochs $(EPOCHS) --train_batch_size $(TRAIN_BATCH_SIZE) --train_num_workers $(TRAIN_NUM_WORKERS) --test_batch_size $(TEST_BATCH_SIZE) --test_num_workers $(TEST_NUM_WORKERS) --lr $(LR) --weight_decay $(WEIGHT_DECAY) --momentum $(MOMENTUM) --verbose 1 --data_download_path $(DATA_DOWNLOAD_PATH) --cuda $(CUDA) --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1

c2: lab2.py
	python lab2.py --optimizer $(OPTIMIZER) --epochs $(EPOCHS) --train_batch_size $(TRAIN_BATCH_SIZE) --train_num_workers $(TRAIN_NUM_WORKERS) --test_batch_size $(TEST_BATCH_SIZE) --test_num_workers $(TEST_NUM_WORKERS) --lr $(LR) --weight_decay $(WEIGHT_DECAY) --momentum $(MOMENTUM) --verbose 0 --data_download_path $(DATA_DOWNLOAD_PATH) --cuda $(CUDA) --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1

c3: tuning.py
	python tuning.py

c4: lab2.py
	python lab2.py --optimizer $(OPTIMIZER) --epochs $(EPOCHS) --train_batch_size $(TRAIN_BATCH_SIZE) --train_num_workers 1 --test_batch_size $(TEST_BATCH_SIZE) --test_num_workers 1 --lr $(LR) --weight_decay $(WEIGHT_DECAY) --momentum $(MOMENTUM) --verbose 0 --data_download_path $(DATA_DOWNLOAD_PATH) --cuda $(CUDA) --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1
	python lab2.py --optimizer $(OPTIMIZER) --epochs $(EPOCHS) --train_batch_size $(TRAIN_BATCH_SIZE) --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size $(TEST_BATCH_SIZE) --test_num_workers $(TEST_NUM_WORKERS) --lr $(LR) --weight_decay $(WEIGHT_DECAY) --momentum $(MOMENTUM) --verbose 0 --data_download_path $(DATA_DOWNLOAD_PATH) --cuda $(CUDA) --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1

c5:
	python lab2.py --optimizer $(OPTIMIZER) --epochs $(EPOCHS) --train_batch_size $(TRAIN_BATCH_SIZE) --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size $(TEST_BATCH_SIZE) --test_num_workers $(TEST_NUM_WORKERS) --lr $(LR) --weight_decay $(WEIGHT_DECAY) --momentum $(MOMENTUM) --verbose 0 --data_download_path $(DATA_DOWNLOAD_PATH) --cuda 0 --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1
	python lab2.py --optimizer $(OPTIMIZER) --epochs $(EPOCHS) --train_batch_size $(TRAIN_BATCH_SIZE) --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size $(TEST_BATCH_SIZE) --test_num_workers $(TEST_NUM_WORKERS) --lr $(LR) --weight_decay $(WEIGHT_DECAY) --momentum $(MOMENTUM) --verbose 0 --data_download_path $(DATA_DOWNLOAD_PATH) --cuda 1 --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1

c6: lab2.py
	python lab2.py --optimizer sgd --epochs 5 --train_batch_size 128 --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size 100 --test_num_workers 2 --lr 0.1 --weight_decay 5e-4 --momentum 0.9 --verbose 0 --data_download_path ./data --cuda 1 --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1
	python lab2.py --optimizer sgd --epochs 5 --train_batch_size 128 --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size 100 --test_num_workers 2 --lr 0.1 --weight_decay 5e-4 --momentum 0.9 --verbose 0 --data_download_path ./data --cuda 1 --nesterov 1 --enable_torch_profiling 0 --include_batch_norm_layers 1
	python lab2.py --optimizer adagrad --epochs 5 --train_batch_size 128 --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size 100 --test_num_workers 2 --lr 0.1 --weight_decay 5e-4 --momentum 0.9 --verbose 0 --data_download_path ./data --cuda 1 --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1
	python lab2.py --optimizer adadelta --epochs 5 --train_batch_size 128 --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size 100 --test_num_workers 2 --lr 0.1 --weight_decay 5e-4 --momentum 0.9 --verbose 0 --data_download_path ./data --cuda 1 --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1
	python lab2.py --optimizer adam --epochs 5 --train_batch_size 128 --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size 100 --test_num_workers 2 --lr 0.1 --weight_decay 5e-4 --momentum 0.9 --verbose 0 --data_download_path ./data --cuda 1 --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 1

c7: lab2.py
	python lab2.py --optimizer sgd --epochs 5 --train_batch_size 128 --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size 100 --test_num_workers 2 --lr 0.1 --weight_decay 5e-4 --momentum 0.9 --verbose 0 --data_download_path ./data --cuda 1 --nesterov 0 --enable_torch_profiling 0 --include_batch_norm_layers 0

ec:
	python lab2.py --optimizer $(OPTIMIZER) --epochs $(EPOCHS) --train_batch_size $(TRAIN_BATCH_SIZE) --train_num_workers $(OPT_NUM_WORKERS) --test_batch_size $(TEST_BATCH_SIZE) --test_num_workers $(TEST_NUM_WORKERS) --lr $(LR) --weight_decay $(WEIGHT_DECAY) --momentum $(MOMENTUM) --verbose 0 --data_download_path $(DATA_DOWNLOAD_PATH) --cuda $(CUDA) --nesterov 0 --enable_torch_profiling 1 --include_batch_norm_layers 1
	tensorboard --logdir=./log