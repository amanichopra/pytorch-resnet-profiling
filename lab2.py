from resnet import ResNet18
import torchvision
import torch
from torch.utils.data import DataLoader
import argparse

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

def train(train_loader, epoch_num, mod, optim, loss_func, device, verbose=False):
    mod.train()
    epoch_loss = 0
    epoch_correct = 0
    epoch_total = 0

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/resnet18'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
    ) as prof:
        for batch_num, (X_batch, y_batch) in enumerate(train_loader):
            if batch_num == 10: break
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            optim.zero_grad()
            out = mod(X_batch)
            loss = loss_func(out, y_batch)
            loss.backward()
            optim.step()
            
            loss = loss.item()
            epoch_loss += loss
            _, pred_labels = out.max(1)
            correct = pred_labels.eq(y_batch).sum().item()
            epoch_correct += correct
            total = out.size(dim=0)
            epoch_total += total

            prof.step()

            if verbose: 
                print(f'Epoch: {epoch_num}, Batch #: {batch_num}, Batch Size: {total}, Training Loss: {loss}, Top-1 Accuracy: {correct/total}')
    
    return epoch_loss, epoch_total, epoch_correct

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ResnetTrain', description='Training script for resnet-18.')
    parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta'], default='sgd', help='Type of optimizer to use in lowercase.')
    parser.add_argument('--epochs', type=int, default=5, help='Number of epochs to train for.')
    parser.add_argument('--train_batch_size', type=int, default=128, help='Batch size used for training data.')
    parser.add_argument('--train_num_workers', type=int, default=2, help='Number of workers used for training dataloader.')
    parser.add_argument('--test_batch_size', type=int, default=100, help='Batch size used for testing data.')
    parser.add_argument('--test_num_workers', type=int, default=2, help='Number of workers used for testing dataloader.')
    parser.add_argument('--verbose', choices=[True, False], type=bool, default=False, help='Whether or not to print debug output.')
    parser.add_argument('--lr', type=float, default=0.1, help='Value for optimizer learning rate.')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Value for optimizer weight decay.')
    parser.add_argument('--momentum', type=float, default=0.9, help='Value for momentum if using SGG or RMSProp optimizers.')
    parser.add_argument('--epsilon', type=float, default=1e-8, help='Value for epsilon if using RMSProp, Adam, Adagrad, or Adadelta.')
    parser.add_argument('--data_download_path', default='./data', help='Path to download CIFAR-10 data.')
    parser.add_argument('--cuda', default=False, type=bool, help='Whether or not to use CUDA. GPUs must be available.')
    parser.add_argument('--nesterov', default=False, type=bool, help='Whether or not to Nesterov momentum if using SGD optimizer.')

    args = parser.parse_args()

    device_name = 'cuda' if args.cuda and torch.cuda.is_available() else 'cpu'
    device = torch.device(device_name)

    if args.verbose: print(f'Using device: {device_name.upper()}.')

    train_loader, test_loader = get_cifar10_dataloaders(args.train_batch_size, args.train_num_workers, args.test_batch_size, args.test_num_workers, download_path=args.data_download_path)

    mod = ResNet18().to(device)

    if args.optimizer == 'sgd':
        optim = torch.optim.SGD(mod.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay, nesterov=args.nesterov)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(mod.parameters(), lr=args.lr, eps=args.epsilson, weight_decay=args.wd)
    elif args.optimizer == 'adagrad':
        optim = torch.optim.Adagrad(mod.parameters(), lr=args.lr, eps=args.epsilson, weight_decay=args.wd)
    elif args.optimizer == 'rmsprop':
        optim = torch.optim.RMSprop(mod.parameters(), lr=args.lr, eps=args.epsilon, momentum=args.momentum, weight_decay=args.weight_decay)
    else:
        optim = torch.optim.Adadelta(mod.parameters(), lr=args.lr, eps=args.epsilson, weight_decay=args.wd)

    loss_func = torch.nn.CrossEntropyLoss()

    losses = []
    total = []
    correct = []

    for i in range(args.epochs):
        epoch_loss, epoch_total, epoch_correct = train(train_loader, i, mod, optim, loss_func, device, verbose=args.epochs)

        





    




    

