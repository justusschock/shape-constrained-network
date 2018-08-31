import os

IMG_EXTENSIONS = [
    ".png", ".PNG", ".jpg", ".JPG"
]

LMK_EXTENSIONS = [
    ".txt", ".TXT"
]

def is_image_file(filename):
    """
    Helper Function to determine whether a file is an image file or not
    :param filename: the filename containing a possible image
    :return: True if file is image file, False otherwise
    """
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def is_landmark_file(filename):
    return any(filename.endswith(extension) for extension in LMK_EXTENSIONS)


def make_dataset(dir):
    """
    Helper Function to make a dataset containing all images in a certain directory
    :param dir: the directory containing the dataset
    :return: images: list of image paths
    """
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images