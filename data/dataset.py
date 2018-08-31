from torch.utils import data as data
from torchvision import transforms
import os
from skimage.transform import resize
from skimage.io import imread
import numpy as np
from .utils import is_image_file, make_dataset, IMG_EXTENSIONS
from menpo import io as mio
import menpo


def menpo_loader(file_name: str, img_size: tuple, crop=True):
    menpo_img = mio.import_image(file_name)

    if crop:
        menpo_img = menpo_img.crop_to_landmarks_proportion(0.1)

    menpo_img = menpo_img.resize(img_size)

    if menpo_img.n_channels > 1:
        menpo_img = menpo_img.as_greyscale()

    img = menpo_img.pixels
    lmk = menpo_img.landmarks[menpo_img.landmarks.group_labels[-1]]
    if menpo.__version__ == '0.7.7':
        points = lmk.lms.points
    else:
        points = lmk.points

    img = img.transpose(1, 2, 0)

    return img, points


def default_loader(file_name: str, img_size: tuple):
    _data = imread(file_name, as_grey=True)

    orig_data_size = _data.shape[:2]
    data_scaling_factors = np.asarray(img_size) / orig_data_size

    _data = resize(_data, img_size, mode='reflect', preserve_range=True)

    # add channels
    if len(_data.shape) < 3:
        _data = _data.reshape((*_data.shape, 1))

    label_file = file_name
    for ext in IMG_EXTENSIONS:
        label_file = label_file.replace(ext, ".txt")

    _label = np.loadtxt(label_file) * data_scaling_factors
    return _data, _label


class ShapeDataset(data.Dataset):
    def __init__(self, data_path, transforms, img_size):
        self.data_path = data_path
        self.transforms = transforms
        self.img_size = img_size

        self.img_files = make_dataset(data_path)

    def __getitem__(self, index):
        # _img, _label = default_loader(self.img_files[index], self.img_size)
        _img, _label = menpo_loader(self.img_files[index], self.img_size)
        if self.transforms is not None:
            _data = (self.transforms(_img), _label)

        else:
            _data = (_img, _label)

        return _data

    def __len__(self):
        return len(self.img_files)


if __name__ == '__main__':
    from matplotlib import pyplot as plt
    dataset = AAMDataset("PATH_TO_DATA_DIR", transforms.ToTensor(), (224, 224))
    data_loader = data.DataLoader(dataset, batch_size=1)

    for idx, tmp in enumerate(data_loader):
        print(idx)
        _data_tensor, _label_tensor = tmp[0], tmp[1]

        plt.imsave(os.path.join("PATH_TO_SAVE_DIR", "FILENAME"), _data_tensor.squeeze().numpy())
