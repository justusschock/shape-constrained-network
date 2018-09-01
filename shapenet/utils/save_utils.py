import torch
import os


def save_network(model, optimizer, epoch, filepath):
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, filepath)


def load_network(filename, model, optimizer=None):
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']))
        return model, optimizer, start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        raise FileNotFoundError("Checkpoint File not found")

