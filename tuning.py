import time
from lab2 import get_cifar10_dataloaders
import torch

TRAIN_BATCH_SIZE = 100
TEST_BATCH_SIZE = 128
DOWNLOAD_PATH = './data'

def get_best_num_workers():
    num_workers = 0
    prev_dl_time = float('inf')

    while True:
        train_loader, _ = get_cifar10_dataloaders(TRAIN_BATCH_SIZE, num_workers, TEST_BATCH_SIZE, num_workers, download_path=DOWNLOAD_PATH)
        train_loader = iter(train_loader)
        torch.cuda.synchronize()
        dl_start = time.perf_counter()
        [batch for batch in train_loader]
        torch.cuda.synchronize()
        dl_end = time.perf_counter()
        dl_time = dl_end - dl_start
        print(f'DL Time for {num_workers} Workers: {dl_time}')

        if prev_dl_time <= dl_time:
            return num_workers - 4

        num_workers += 4
        prev_dl_time = dl_time

if __name__ == '__main__':
    best_num_workers = get_best_num_workers()
    print(f'{best_num_workers} Workers Needed for Best Runtime Performance!')