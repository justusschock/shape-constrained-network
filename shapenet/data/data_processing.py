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
    """
    Holds Single Image
    """
    def __init__(self, img, lmk, **kwargs):
        """

        Parameters
        ----------
        img: np.ndarray
            actual image pixels
        lmk: np.ndarray
            landmarks
        kwargs: dict
            additional kwargs like file paths
        """
        self.img = img
        self.lmk = lmk
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_files(cls, file):
        """
        Create class from image or landmark file
        Parameters
        ----------
        file: string
            path to image or landmarkfile

        Returns
        -------
        class instance
        """
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
        """
        Creates class from menpo Image
        Parameters
        ----------
        menpo_img: menpo.image.Image
            menpo image to create class from
        kwargs: dict
            additional keyword arguments

        Returns
        -------
        class instance
        """
        lmk = menpo_img.landmarks[menpo_img.landmarks.group_labels[-1]]
        if menpo.__version__ == '0.7.7':
            points = lmk.lms.points
        else:
            points = lmk.points

        img = menpo_img.pixels

        return cls(img, points, **kwargs)

    def resize(self, img_size):
        """
        Resize Image

        Parameters
        ----------
        img_size: new image size

        Returns
        -------
        resized image
        """
        return self.__class__.from_menpo(self.as_menpo_img().resize(img_size))

    @property
    def bbox(self):
        """
        Compute bounding box

        Returns
        -------
        tuple: bounding box y_min, x_min, y_max, x_max

        """
        transposed_lmk = self.lmk.transpose()
        y_min, x_min = transposed_lmk.min(axis=1)
        y_max, x_max = transposed_lmk.max(axis=1)
        return y_min, x_min, y_max, x_max

    @property
    def img_size(self):
        """
        Returns current image size

        Returns
        -------
        tuple: img_size
        """
        try:
            img_size = self.img.shape[:1]
        except:
            __tmp = self.bbox
            img_size = (__tmp[2] - __tmp[0], __tmp[3] - __tmp[1])

        return img_size

    def as_menpo_img(self):
        """
        Converts image to menpo image

        Returns
        -------
        menpo.image.Image: current image as menpo image
        """
        menpo_img = Image(self.img)
        lmk = self.lmk
        menpo_lmk = PointCloud(self.lmk)
        lmk_manager = LandmarkManager()
        lmk_manager["LMK"] = menpo_lmk
        menpo_img.landmarks = lmk_manager

        return menpo_img

    def view(self, **kwargs):
        """
        Show image (with landmarks if available)

        Parameters
        ----------
        kwargs: dict
            additional keyword arguments
        """
        try:
            self.as_menpo_img().view_landmarks(**kwargs)
        except:
            self.as_menpo_img().view(**kwargs)


class DataProcessing(object):
    """
    Process multiple SingleImages
    """
    def __init__(self, samples, **kwargs):
        """

        Parameters
        ----------
        samples: list
            list of SingleImages
        kwargs: dict
            additional keyword arguments
        """
        super().__init__()
        self.samples = samples
        for key, val in kwargs.items():
            setattr(self, key, val)

    @classmethod
    def from_dir(cls, data_dir, verbose=True):
        """
        create class instance from directory

        Parameters
        ----------
        data_dir: string
            directory where data is stored
        verbose: bool
            whether or not to print current progress

        Returns
        -------
        class instance
        """

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
        """
        Create class instance from directory of menpo files

        Parameters
        ----------
        data_dir: string
            path to data directory
        verbose: bool
            whether or not to print current progress

        Returns
        -------
        class instance
        """
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
        """
        get list of samples' landmarks

        Returns
        -------
        list: landmarks
        """
        return [tmp.lmk for tmp in self.samples]

    @property
    def images(self):
        """
        get list of samples' pixels

        Returns
        -------
        list: pixels
        """
        return [tmp.img for tmp in self.samples]

    def resize(self, img_size):
        """
        resize all samples

        Parameters
        ----------
        img_size: tuple
            new image size
        """
        for idx, sample in enumerate(self.samples):
            self.samples[idx] = sample.resize(img_size)

    @staticmethod
    def _get_files(directory, extensions):
        """
        return files with extensions

        Parameters
        ----------
        directory: string
            directory containing the files
        extensions: list
            list of strings specifying valid extensions

        Returns
        -------
        list: valid files
        """

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
        """
        perform PCA on samples' landmarks

        Parameters
        ----------
        scale: bool
            whether or not to scale the principa components with the corresponding eigen value
        center: bool
            whether or not to substract mean before pca
        args: list
            additional positional arguments (passed to pca)
        kwargs: dict
            additional keyword arguments (passed to pca)

        Returns
        -------
        np.array: eigen_shapes
        """
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

