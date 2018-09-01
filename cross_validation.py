import os


import torch
import menpo
from torch.utils.data import ConcatDataset, DataLoader
from sklearn.model_selection import KFold
from data import AAMDataset, DataProcessing, IMG_EXTENSIONS
from models import ShapeNetwork
from menpo import io as mio
from menpo.shape import PointCloud
from menpo.transform import TransformChain
from tensorboardX import SummaryWriter
from utils import save_network, load_network
from torchvision.transforms import Compose, ToTensor, Normalize
import shutil
from menpo.landmark import LandmarkManager

torch.cuda.benchmark = True


def kfold_cross_validation(data_paths, model_cls=ShapeNetwork, img_size=224, batch_size=1, num_epochs=50, lr=1e-4,
                           out_path=".", loss_fn=torch.nn.MSELoss(), n_shape_params=None):
    """
    Cross Validation

    Parameters
    ----------
    data_paths: list
        paths containing data
    model_cls: Any
        model Class to train
    img_size: int
        image size
    batch_size: int
        batch size
    num_epochs: int
        number of epochs
    lr: float
        learning rate
    out_path: string
        path to save results and weights to
    loss_fn: torch.nn.Module
        loss function
    n_shape_params: int
        number of shape parameters

    """

    outdir = os.path.join(out_path, "%d_epochs" % num_epochs)
    train_transforms = Compose([ToTensor(), Normalize([0], [1])])
    val_transforms = Compose([ToTensor(), Normalize([0], [1])])

    kfold = KFold(n_splits=len(data_paths), random_state=None, shuffle=False)

    for train_paths, test_path in kfold.split(data_paths):

        best_val_los = float('inf')

        curr_out_path = os.path.join(outdir, os.path.split(data_paths[test_path])[-1])
        _trainsets = []
        for idx in train_paths:
            _trainsets.append(AAMDataset(data_paths[idx], train_transforms, (img_size, img_size)))

        trainset = ConcatDataset(_trainsets)
        train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=4)

        valset = AAMDataset(data_paths[test_path], val_transforms, (img_size, img_size))
        val_loader = DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=4)

        pca_data = DataProcessing.from_menpo(data_paths[train_paths[0]])

        lmk_pca = pca_data.lmk_pca(scale=True, center=True, n_components=n_shape_params)

        model = model_cls(lmk_pca, len(lmk_pca)-1, 1, img_size=img_size, in_channels=1,
                          verbose=False, norm_type='instance')

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")

        optim = torch.optim.Adam(model.parameters(), lr=lr)

        model = model.to(device).train()

        writer = SummaryWriter(curr_out_path)

        checkpoint_dir = os.path.join(curr_out_path, "checkpoints")
        pred_dir = os.path.join(curr_out_path, "preds")

        os.makedirs(checkpoint_dir, exist_ok=True)
        os.makedirs(pred_dir, exist_ok=True)

        for epoch in range(1, num_epochs+1):
            print("Epoch %03d of %03d" % (epoch, num_epochs))
            model = model.train()
            optim, model = model.single_epoch(train_loader, optim, model, loss_fn, writer, device=device)

            model = model.eval()
            curr_val_loss = model.validate(val_loader, model, loss_fn, writer, device=device)

            save_network(model, optim, epoch, os.path.join(checkpoint_dir, "model_epoch_%d.pth" % epoch))

            if curr_val_loss <= best_val_los:
                print("New Best Model Found. Congrats!")
                shutil.copy(os.path.join(checkpoint_dir, "model_epoch_%d.pth" % epoch),
                            os.path.join(checkpoint_dir, "model_best.pth"))
                best_val_los = curr_val_loss

        model, _, _ = load_network(os.path.join(checkpoint_dir, "model_best.pth"), model.cpu(), optim)
        model = model.to(device)

        model.eval()
        val_data = DataProcessing.from_menpo(data_paths[test_path])

        for i in range(len(val_data)):
            _data = val_data[i]
            _menpo_img = _data.as_menpo_img()

            _cropped_img, trafo_crop = _menpo_img.crop_to_landmarks_proportion(0.1, return_transform=True)
            _cropped_img, trafo_resize = _cropped_img.resize((img_size, img_size), return_transform=True)
            _path = _data.img_file

            applied_trafos = TransformChain([trafo_resize, trafo_crop])

            img = val_transforms(_cropped_img.pixels.transpose(1, 2, 0))

            img = img.to(device).to(torch.float)

            _pred = model(img.unsqueeze(0))

            _pred_file = os.path.join(pred_dir, os.path.split(str(_path))[-1])
            _pred_file = _pred_file.rsplit(".", maxsplit=1)[0]

            gt_lmk = _menpo_img.landmarks[_menpo_img.landmarks.group_labels[-1]]

            mio.export_image(_menpo_img, _pred_file + ".png", overwrite=True)
            mio.export_landmark_file(gt_lmk, _pred_file + "_gt.ljson", overwrite=True)
            _lmk_man = LandmarkManager()

            remapped_pred = applied_trafos.apply(PointCloud(_pred.cpu().detach()[0].numpy().squeeze()))
            _lmk_man["pred"] = remapped_pred
            mio.export_landmark_file(_lmk_man["pred"], _pred_file + "_pred.ljson", overwrite=True)


if __name__ == '__main__':

    DATA_DIRS = []
    OUT_PATH = ""


    kfold_cross_validation(data_paths=DATA_DIRS, num_epochs=2, out_path=OUT_PATH)