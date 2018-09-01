from setuptools import setup
import os


# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="shapenet",
    version="0.1.0",
    author="Justus Schock",
    author_email=("Implementation of shape-constrained networks "
                  "and necessary data processing"),
    license="MIT",
    keywords="pytorch deep learning shape network",
    url="https://github.com/justusschock/shape-constrained-network.git",
    packages=["shapenet"],
    long_description=read('README'),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Programming Language :: C++",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ], install_requires=['torch', 'numpy', 'menpo', 'torchvision']
)