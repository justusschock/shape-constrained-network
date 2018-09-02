import torch
import os


def save_network(model, optimizer, epoch, filepath):
    """
    Saves state dict of model, optimizer, current epoch

    Parameters
    ----------
    model: torch.nn.Module
        model whose state_dict should be saved
    optimizer: torch.optim.Optimizer
        optimizer whose state_dict should be saved
    epoch: int
        current epoch
    filepath: string
        file to save the state_dicts to

    Returns
    -------

    """
    state = {
        'epoch': epoch,
        'state_dict': model.state_dict(),
        'optimizer' : optimizer.state_dict(),
    }
    torch.save(state, filepath)


def load_network(filename, model, optimizer=None, **kwargs):
    """
    Loads state_dicts to model and optimizer

    Parameters
    ----------
    filename: string
        file to load from
    model: torch.nn.Module
        modle to load state_dict to
    optimizer: torch.optim.Optimizer or None
        if not None: optimizer to load state_dict to
    kwargs:
        additional keyword arguments (directly passed to torch.load)

    Returns
    -------
    model, optimizer and start epoch
    """
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename, **kwargs)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['state_dict'])
        if optimizer is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(filename, checkpoint['epoch']+1))
        return model, optimizer, start_epoch
    else:
        print("=> no checkpoint found at '{}'".format(filename))
        raise FileNotFoundError("Checkpoint File not found")

