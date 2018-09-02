import os
import torch
import numpy as np
from menpo import io as mio
from menpo.landmark import LandmarkManager
from menpo.shape import PointCloud
from menpo.transform import TransformChain
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
from .data import DataProcessing
from .utils import load_network, NetConfig
from .models import ShapeNetwork
from matplotlib import pyplot as plt

def predict(data_path, config: NetConfig, weight_path, output_path=None,
            gts=True, transforms=Compose([ToTensor(), Normalize([0], [1])]),
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")):

    model = ShapeNetwork(np.array([np.zeros((68, 2))
                                   for i in range(config.num_shape_params+1)]),
                         config.num_shape_params, config.num_global_params,
                         config.num_translation_params, config.img_size,
                         norm_type=config.norm, use_cpp=config.use_cpp
                         )

    if output_path is None:
        output_path = os.path.join(data_path, "preds_shapenet")

    os.makedirs(output_path, exist_ok=True)

    model, _, _ = load_network(weight_path, model, None, map_location="cpu")

    model = model.to(device)

    data = DataProcessing.from_menpo(data_path)

    model.eval()

    print("Start Prediction")
    for i in tqdm(range(len(data))):
        _data = data[i]
        _menpo_img = _data.as_menpo_img()

        if _menpo_img.n_channels > 1:
            _menpo_img = _menpo_img.as_greyscale()

        _cropped_img, trafo_crop = _menpo_img.crop_to_landmarks_proportion(0.1,
                                                                           return_transform=True)
        _cropped_img, trafo_resize = _cropped_img.resize((config.img_size, config.img_size),
                                                         return_transform=True)
        _path = _data.img_file

        applied_trafos = TransformChain([trafo_resize, trafo_crop])

        img = transforms(_cropped_img.pixels.transpose(1, 2, 0))

        img = img.to(device).to(torch.float)

        _pred = model(img.unsqueeze(0))

        _pred_file = os.path.join(output_path, os.path.split(str(_path))[-1])
        _pred_file = _pred_file.rsplit(".", maxsplit=1)[0]

        _menpo_img.landmarks["pred"] = applied_trafos.apply(
            PointCloud(_pred.cpu().detach()[0].numpy().squeeze())
        )

        plt.figure()
        tmp = _menpo_img.view_landmarks(group="LMK", marker_face_colour='r', marker_edge_colour='r')
        try:
            tmp.save_figure(_pred_file + "_gt.png", bbox_inches='tight')
        except TypeError:
            tmp.save_figure(_pred_file + "_gt.png")
        plt.close()
        plt.figure()
        tmp = _menpo_img.view_landmarks(group="pred", marker_face_colour='r', marker_edge_colour='r')
        try:
            tmp.save_figure(_pred_file + "_pred.png", bbox_inches='tight')
        except TypeError:
            tmp.save_figure(_pred_file + "_pred.png")
        plt.close()


if __name__ == '__main__':
    print("Start Prediction")
    config = NetConfig("/home/schock/Downloads/config.yaml", "default")
    DATA_PATHS = ["/home/schock/Downloads/300W/01_Indoor", "/home/schock/Downloads/300W/02_Outdoor"]
    for path in DATA_PATHS:
        predict(path, config,
                "/home/schock/Downloads/model_epoch_70.pth",
                )
