import os

import matplotlib

from shapenet.data import DataProcessing, ShapeDataset
from shapenet.utils import save_network
import argparse
import sys
from shapenet.utils.net_config import NetConfig
from shapenet.models import AbstractNetwork, ShapeNetwork
from datetime import datetime
import shutil
import torch
from torch.utils.data import DataLoader
import numpy as np
from tensorboardX import SummaryWriter

from torchvision.transforms import ToTensor, Compose, Normalize


def adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs):
    """
    implements simple linear lr scheduling

    Parameters
    ----------
    optimizer: torch.optim.Optimizer
        the optimizer yielding the current learning rate
    epoch: int
        current epoch of training
    initial_lr: float
        Initial learning rate at beginning of training
    num_epochs: int
        total number of epochs to train
    """
    decay = initial_lr / num_epochs
    lr = initial_lr - decay*epoch
    print("Set LR to %f" % lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def train(train_data_path, validation_data_path, num_epochs, out_path, loss_fn,
          model: AbstractNetwork, optimizer: torch.optim.Optimizer,
          batch_size=1, verbose=True, initial_lr=1e-3,
          transformations={"train": Compose([ToTensor()]),
                           "validate": Compose([ToTensor()])}, **kwargs):
    """
    Implements a basic train routine by calling model's single epoch and
    validate functions

    Parameters
    ----------
    train_data_path
    validation_data_path
    num_epochs
    out_path
    loss_fn
    model
    optimizer
    batch_size
    verbose
    initial_lr
    transformations
    kwargs

    Returns
    -------

    """

    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    model = model.to(device)

    img_size = (224, 224)
    trainset = ShapeDataset(train_data_path, transformations["train"],
                            img_size)
    validation_set = ShapeDataset(validation_data_path,
                                  transformations["validate"], img_size)

    if sys.platform == 'win32':
        num_workers = 0
    else:
        num_workers = 4

    train_loader = DataLoader(trainset, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers)
    validation_loader = DataLoader(validation_set, batch_size=batch_size,
                                   shuffle=False, num_workers=num_workers)

    translation_params = torch.from_numpy(np.asarray(
        img_size)/2).float().to(device)

    os.makedirs(os.path.join(out_path, "Checkpoints"), exist_ok=True)

    writer = SummaryWriter(out_path)

    for epoch in range(num_epochs):
        print("Epoch %03d of %03d" % (epoch+1, num_epochs))
        curr_out_dir = os.path.join(out_path, "Epoch_%03d" % (epoch+1))
        optimizer, model = model.single_epoch(
            train_loader, optimizer, model, loss_fn,
            device=device, verbose=verbose, img_size=min(img_size),
            curr_epoch=epoch, writer=writer, **kwargs)

        with torch.no_grad():
            print("\tValidating")
            model.validate(validation_loader, model, loss_fn,
                           device=device, verbose=verbose,
                           img_size=min(img_size), curr_epoch=epoch,
                           out_dir=curr_out_dir, writer=writer, **kwargs)

        save_network(model, optimizer, epoch, os.path.join(
            out_path, "Checkpoints", "model_epoch_%d.pth" % (epoch+1)))

        adjust_learning_rate(optimizer, epoch, initial_lr, num_epochs)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Select User and Configuration file")
    parser.add_argument("-u", "--user", help="Select User", dest="user",
                        type=str, default="default")
    parser.add_argument("-f", "--config_file",
                        help="Select Configuration file", dest="config_file",
                        type=str, default="./config.yaml")

    yaml_file_path = os.path.join(os.path.split(
        os.path.realpath(__file__))[0], "config.yaml")

    parse_args = sys.argv[1:] if len(sys.argv) > 1 else ["-u", "default", "-f",
                                                         yaml_file_path]
    parsed_args = parser.parse_args(parse_args)

    config = NetConfig(parsed_args.config_file, parsed_args.user)

    eigen_shapes = DataProcessing.from_menpo(config.pca_data_path).lmk_pca(
        config.scale, config.center, n_components=config.num_shape_params)

    model = ShapeNetwork(eigen_shapes, config.num_shape_params,
                         config.num_global_params,
                         config.num_translation_params, config.img_size,
                         norm_type=config.norm, use_cpp=config.use_cpp)


    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=float(config.initial_lr))

    out_path = os.path.join(config.output_path, config.network_type,
                            str(datetime.now().strftime("%y-%m-%d_%H-%M-%S")))
    os.makedirs(out_path, exist_ok=True)
    shutil.copy2(parsed_args.config_file, out_path)

    transformations = {"train": Compose([ToTensor(), Normalize([0], [1])]),
                       "validate": Compose([ToTensor(), Normalize([0], [1])])
                       }

    train(
        config.network_train_path,
        config.validation_path,
        config.num_epochs,
        out_path,
        torch.nn.MSELoss(),
        model,
        optimizer,
        config.batch_size,
        config.verbose,
        float(config.initial_lr),
        transformations=transformations,
        save_outputs=False
    )
