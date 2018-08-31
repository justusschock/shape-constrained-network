import os
from sklearn.decomposition import PCA
import skimage
import menpo
from skimage import io as sio
from skimage import transform
import numpy as np
import glob
from menpo import io as mio
from .utils import is_landmark_file, is_image_file, LMK_EXTENSIONS, IMG_EXTENSIONS
from matplotlib import pyplot as plt
from menpo.image import Image
from menpo.landmark import LandmarkManager
from menpo.shape import PointCloud
from tqdm import tqdm


class SingleImage(object):
    def __init__(self, img, lmk, **kwargs):
        self.img = img
        self.lmk = lmk
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_files(cls, file):
        is_img_file = is_image_file(file)
        is_lmk_file = is_landmark_file(file)

        img, lmk = None, None

        if is_img_file:
            img = sio.imread(file)
            img_file = file,
            lmk_file = None
            for ext in LMK_EXTENSIONS:
                curr_ext = "." + file.rsplit(".", maxsplit=1)[-1]

                _lmk_file = file.replace(curr_ext, ext)
                if os.path.isfile(_lmk_file):
                    lmk = np.loadtxt(_lmk_file)
                    lmk_file = _lmk_file

        elif is_lmk_file:
            lmk = np.loadtxt(file)
            lmk_file = file
            img_file = None

            for ext in IMG_EXTENSIONS:
                curr_ext = "." + file.rsplit(".", maxsplit=1)[-1]

                _img_file = file.replace(curr_ext, ext)

                if os.path.isfile(_img_file):
                    img = sio.imread(_img_file)
                    img_file = _img_file

        else:
            raise FileNotFoundError("No Suitable File given")

        return cls(img, lmk, img_file=img_file, lmk_file=lmk_file)

    @classmethod
    def from_menpo(cls, menpo_img: Image, **kwargs):
        lmk = menpo_img.landmarks[menpo_img.landmarks.group_labels[-1]]
        if menpo.__version__ == '0.7.7':
            points = lmk.lms.points
        else:
            points = lmk.points

        img = menpo_img.pixels

        return cls(img, points, **kwargs)

    def resize(self, img_size):
        _curr_size = self.img_size
        _scales = np.asarray(img_size) / np.asarray(_curr_size)
        self.img = transform.resize(self.img, img_size)
        self.lmk = self.lmk * _scales

    @property
    def bbox(self):
        transposed_lmk = self.lmk.transpose()
        y_min, x_min = transposed_lmk.min(axis=1)
        y_max, x_max = transposed_lmk.max(axis=1)
        return y_min, x_min, y_max, x_max

    @property
    def img_size(self):
        try:
            img_size = self.img.shape[:1]
        except:
            __tmp = self.bbox
            img_size = (__tmp[2] - __tmp[0], __tmp[3] - __tmp[1])

        return img_size

    def as_menpo_img(self):
        menpo_img = Image(self.img)
        lmk = self.lmk
        menpo_lmk = PointCloud(self.lmk)
        lmk_manager = LandmarkManager()
        lmk_manager["LMK"] = menpo_lmk
        menpo_img.landmarks = lmk_manager

        return menpo_img

    def view(self, **kwargs):
        self.as_menpo_img().view_landmarks(**kwargs)


class DataProcessing(object):
    def __init__(self, samples, **kwargs):
        super().__init__()
        self.samples = samples
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_dir(cls, data_dir, verbose=True):

        if verbose:
            print("Loading data from %s" % data_dir)
            wrapper_fn = tqdm
        else:
            def linear_wrapper(x):
                return x

            wrapper_fn = linear_wrapper

        files = cls._get_files(data_dir, IMG_EXTENSIONS)
        samples = []
        for file in wrapper_fn(files):
            samples.append(SingleImage.from_files(file))

        return cls(samples=samples)

    @classmethod
    def from_menpo(cls, data_dir, verbose=True):
        if verbose:
            print("Loading data from %s" % data_dir)
            wrapper_fn = tqdm
        else:
            def linear_wrapper(x):
                return x

            wrapper_fn = linear_wrapper
        img_paths = list(mio.image_paths(data_dir))

        samples = []
        for _img in wrapper_fn(img_paths):

            samples.append(SingleImage.from_menpo(mio.import_image(_img), img_file=_img))

        return cls(samples=samples)

    @property
    def landmarks(self):
        return [tmp.lmk for tmp in self.samples]

    @property
    def images(self):
        return [tmp.img for tmp in self.samples]

    def resize(self, img_size):
        for idx, sample in enumerate(self.samples):
            self.samples[idx] = sample.resize(img_size)

    @staticmethod
    def _get_files(directory, extensions):

        files = []

        if not isinstance(extensions, list):
            extensions = [extensions]
        for ext in extensions:
            ext = ext.strip(".")
            files += glob.glob(directory + "/*." + ext)
        files.sort()
        return files

    def __getitem__(self, index):
        return self.samples[index]

    def __len__(self):
        return len(self.samples)

    def lmk_pca(self, scale: bool, center: bool, *args, **kwargs):
        landmarks = np.asarray(self.landmarks)
        if center:
            mean = np.mean(landmarks.reshape(-1, 2), axis=0)
            landmarks = landmarks - mean
        landmarks_transposed = landmarks.transpose((0, 2, 1))

        reshaped = landmarks_transposed.reshape(landmarks.shape[0], -1)
        pca = PCA(*args, **kwargs)
        pca.fit(reshaped)

        if scale:
            components = pca.components_ * pca.singular_values_.reshape(-1, 1)
        else:
            components = pca.components_

        return np.array([pca.mean_] + list(components)).reshape(components.shape[0] + 1,
                                                                *landmarks_transposed.shape[1:]).transpose(0, 2, 1)

