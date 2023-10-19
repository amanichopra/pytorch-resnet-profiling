# PyTorch Resnet Profiling

This repo contains a Makefile to profile Resnet-18 (trained on CIFAR-10) on dataloading, training, I/O, and inference. The following infrastructure was used: The following hardware was used: GCP n1-standard-8 VM w/ 1 Tesla P4 GPU running [this](http://pytorch-2-0-gpu-v20230822-debian-11-py310) container with 100GB boot disk. Run the profile module as follows:
1. ```make c2```: reports dataloading, training, and total running time for each epoch.
2. ```make c3```: tunes the number of workers hyperparameter for dataloading; reports dataloading time.
3. ```make c4```: benchmarks dataloading using 1 worker vs. the optimal number of workers found in ```make c3```.
4. ```make c5```: benchmarks training time in CPU vs. GPU.
5. ```make c6```: benchmarks training loss and accuracy across different optimizers (SGD, SGD w/ Nesterov, Adagrad, Adadelta, and Adam).
6. ```make c7```: benchmarks training loss and accuracy in Resnet-18 with and without batch normalization layers.