from resnet import ResNet18
import torch
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='ResnetTrain', description='Training script for resnet-18.')
    parser.add_argument('--optimizer', choices=['sgd', 'adam', 'adagrad', 'rmsprop', 'adadelta'], default='sgd', help='Type of optimizer to use in lowercase.')
    args = parser.parse_args()

    model = ResNet18()
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad) # not dependent on optimizer
    
    # need to do one forward and backward pass to get gradients
    input = torch.ones(1, 3, 32, 32)
    ground_truth_labs = torch.ones(1).long()
    preds = model(input)

    loss_func = torch.nn.CrossEntropyLoss()
    loss = loss_func(preds, ground_truth_labs)

    if args.optimizer == 'sgd':
        optim = torch.optim.SGD(model.parameters(), lr=0.1)
    elif args.optimizer == 'adam':
        optim = torch.optim.Adam(model.parameters())
    elif args.optimizer == 'adagrad':
        optim = torch.optim.Adagrad(model.parameters())
    elif args.optimizer == 'rmsprop':
        optim = torch.optim.RMSprop(model.parameters())
    else:
        optim = torch.optim.Adadelta(model.parameters())

    optim.zero_grad()
    loss.backward()
    optim.step()

    num_grads = sum(p.grad.numel() for p in model.parameters() if p.requires_grad)
    print(f'Optimizer: {args.optimizer}, # Trainable Params: {num_params}, # Gradients: {num_grads}')