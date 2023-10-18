from resnet import ResNet18
import torchvision
import torch
from torch.utils.data import DataLoader
import argparse
import time

def get_cifar10_dataloaders(train_batch_size, train_num_workers, test_batch_size, test_num_workers, download_path):
    train_transforms = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop(32, padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    test_transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])
    ])

    train = torchvision.datasets.CIFAR10(root=download_path, train=True, download=True, transform=train_transforms)
    train_loader = DataLoader(train, batch_size=train_batch_size, num_workers=train_num_workers)
    test = torchvision.datasets.CIFAR10(root=download_path, train=False, download=True, transform=test_transforms)
    test_loader = DataLoader(test, batch_size=test_batch_size, num_workers=test_num_workers)

    return train_loader, test_loader

def train(train_loader, epoch_num, mod, optim, loss_func, device, profile, verbose=False):
    mod.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0
    epoch_dl_time = 0
    epoch_train_time = 0
    epoch_metrics_time = 0
    n_batches = len(train_loader)
    n_samples = len(train_loader.dataset)
    train_loader = iter(train_loader)

    if profile:
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
            record_shapes=True,
            profile_memory=True,
            with_stack=True
        ) as prof:
            for batch_num in range(n_batches):
                torch.cuda.synchronize()
                dl_start = time.perf_counter()
                X_batch, y_batch = next(train_loader)
                torch.cuda.synchronize()
                dl_end = time.perf_counter()
                dl_time = dl_end - dl_start
                epoch_dl_time += dl_time

                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)

                torch.cuda.synchronize()
                train_start = time.perf_counter()
                optim.zero_grad()
                out = mod(X_batch)
                loss = loss_func(out, y_batch)
                loss.backward()
                optim.step()
                torch.cuda.synchronize()
                train_end = time.perf_counter()
                train_time = train_end - train_start
                epoch_train_time += train_time
                
                torch.cuda.synchronize()
                metrics_start = time.perf_counter()
                loss = loss.item()
                epoch_loss += loss * X_batch.size(0)
                _, pred_labels = out.max(1)
                correct = pred_labels.eq(y_batch).sum().item()
                epoch_correct += correct
                total = out.size(dim=0)
                epoch_total += total
                torch.cuda.synchronize()
                metrics_end = time.perf_counter()
                metrics_time = metrics_end - metrics_start
                epoch_metrics_time += metrics_time
                
                prof.step()

                if verbose: 
                    print(f'Epoch: {epoch_num + 1}, Batch #: {batch_num + 1}, Batch Size: {total}, Training Loss: {loss}, Top-1 Accuracy: {correct/total}')
        
    else:
        for batch_num in range(n_batches):
            torch.cuda.synchronize()
            dl_start = time.perf_counter()
            X_batch, y_batch = next(train_loader)
            torch.cuda.synchronize()
            dl_end = time.perf_counter()
            dl_time = dl_end - dl_start
            epoch_dl_time += dl_time

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            torch.cuda.synchronize()
            train_start = time.perf_counter()
            optim.zero_grad()
            out = mod(X_batch)
            loss = loss_func(out, y_batch)
            loss.backward()
            optim.step()
            torch.cuda.synchronize()
            train_end = time.perf_counter()
            train_time = train_end - train_start
            epoch_train_time += train_time
            
            torch.cuda.synchronize()
            metrics_start = time.perf_counter()
            loss = loss.item()
            epoch_loss += loss * X_batch.size(0)
            _, pred_labels = out.max(1)
            correct = pred_labels.eq(y_batch).sum().item()
            epoch_correct += correct
            total = out.size(dim=0)
            epoch_total += total
            torch.cuda.synchronize()
            metrics_end = time.perf_counter()
            metrics_time = metrics_end - metrics_start
            epoch_metrics_time += metrics_time

            if verbose: 
                print(f'Epoch: {epoch_num + 1}, Batch #: {batch_num + 1}, Batch Size: {total}, Training Loss: {loss}, Top-1 Accuracy: {correct/total}')

    epoch_loss /= n_samples 

    return epoch_loss, epoch_total, epoch_correct, epoch_dl_time, epoch_train_time, epoch_metrics_time 

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ResnetTrain', description='Training script for resnet-18.')
    parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta'], default='sgd', help='Type of optimizer to use in lowercase.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size used for training data.')
    parser.add_argument('--train_num_workers', type=int, default=2, help='Number of workers used for training dataloader.')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size used for testing data.')
    parser.add_argument('--test_num_workers', type=int, default=2, help='Number of workers used for testing dataloader.')
    parser.add_argument('--verbose', type=int, default=0, help='Whether or not to print debug output.')
    parser.add_argument('--lr', type=float, default=0.1, help='Value for optimizer learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Value for optimizer weight decay.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Value for momentum if using SGG or RMSProp optimizers.')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Value for epsilon if using RMSProp, Adam, Adagrad, or Adadelta.')
    parser.add_argument('--data_download_path', default='./data', help='Path to download CIFAR-10 data.')
    parser.add_argument('--cuda', default=0, type=int, help='Whether or not to use CUDA. GPUs must be available.')
    parser.add_argument('--nesterov', default=0, type=int, help='Whether or not to Nesterov momentum if using SGD optimizer.')
    parser.add_argument('--enable_torch_profiling', default=0, type=int, help='Whether or not to enable the Pytorch profiler during training.')
    parser.add_argument('--include_batch_norm_layers', default=1, type=int, help='Whether or not to include batch norm layers in ResNet model.') 

    args = parser.parse_args()

    device_name = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    if args.verbose: print(f'Using device: {device_name.upper()}.')

    train_loader, test_loader = get_cifar10_dataloaders(args.train_batch_size, args.train_num_workers, args.test_batch_size, args.test_num_workers, download_path=args.data_download_path)

    mod = ResNet18(include_batch_norm_layers=args.include_batch_norm_layers).to(device)

    if args.optimizer == 'sgd':
        optim = torch.optim.SGD(mod.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(mod.parameters(), lr=args.lr, eps=args.epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == 'adagrad':
        optim = torch.optim.Adagrad(mod.parameters(), lr=args.lr, eps=args.epsilon, weight_decay=args.weight_decay)
    elif args.optimizer == 'rmsprop':
        optim = torch.optim.RMSprop(mod.parameters(), lr=args.lr, eps=args.epsilon, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.Adadelta(mod.parameters(), lr=args.lr, eps=args.epsilon, weight_decay=args.weight_decay)

    loss_func = torch.nn.CrossEntropyLoss()

    dl_times = []
    train_times = []
    metrics_times = []
    epoch_times = []

    for i in range(args.epochs):
        torch.cuda.synchronize()
        epoch_start_time = time.perf_counter()
        epoch_loss, epoch_total, epoch_correct, epoch_dl_time, epoch_train_time, epoch_metrics_time = train(train_loader, i, mod, optim, loss_func, device, args.enable_torch_profiling, verbose=args.verbose)
        dl_times.append(epoch_dl_time)
        train_times.append(epoch_train_time)
        metrics_times.append(epoch_metrics_time)
        torch.cuda.synchronize()
        epoch_end_time = time.perf_counter()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_times.append(epoch_time)

        print(f'[EPOCH {i+1} SUMMARY] Total/AVG DL Time: {epoch_dl_time}/{epoch_dl_time / len(train_loader)}, Total/AVG Train Time: {epoch_train_time}/{epoch_train_time / len(train_loader)}, Total/AVG Metrics Time: {epoch_metrics_time}/{epoch_metrics_time / len(train_loader)}, Total/AVG Running Time: {epoch_time}/{epoch_time / len(train_loader)}, Training Loss: {epoch_loss}, Top-1 Accuracy: {epoch_correct / epoch_total}')
        print()
    
    dl_time = sum(dl_times)
    avg_dl_time = dl_time / args.epochs
    train_time = sum(train_times)
    avg_train_time = train_time / args.epochs
    metrics_time = sum(metrics_times)
    avg_metrics_time = metrics_time / args.epochs
    runtime = sum(epoch_times)
    avg_runtime = runtime / args.epochs

    print(f'[BENCHMARKING SUMMARY ACROSS ALL EPOCHS] Total/AVG DL Time: {dl_time}/{avg_dl_time}, Total/AVG Train Time: {train_time}/{avg_train_time}, Total/AVG Metrics Time: {metrics_time}/{avg_metrics_time}, Total/AVG Runtime Across All Epochs: {runtime}/{avg_runtime}')
        





    




    

