from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import subprocess
import setuptools
from setuptools import dist
from distutils.core import setup, Extension

dist.Distribution().fetch_build_eggs(["Cython", "numpy"])
import numpy as np

cwd = os.path.dirname(os.path.abspath(__file__))
subprocess.check_output(["bash", "models/graph/cmu/download.sh"], cwd=cwd)
POSE_DIR = os.path.realpath(os.path.dirname(__file__))

REQUIRED_PACKAGES = [
    "argparse>=1.1",
    "dill==0.2.7.1",
    "fire >= 0.1.3",
    "matplotlib >= 2.2.2",
    "psutil >= 5.4.5",
    "requests >= 2.18.4",
    "scikit-image >= 0.13.1",
    "scipy >= 1.1.0",
    "slidingwindow >= 0.0.13",
    "opencv-python",
    "tqdm >= 4.23.4",
    "tensorflow < 2.0",
    "tensorpack >= 0.8.5",
    "pycocotools",
]

DEPENDENCY_LINKS = [
    "git+https://github.com/ppwwyyxx/tensorpack.git#egg=tensorpack",
]

EXT = Extension(
    "_pafprocess",
    sources=[
        "tf_pose/pafprocess/pafprocess_wrap.cpp",
        "tf_pose/pafprocess/pafprocess.cpp",
    ],
    swig_opts=["-c++"],
    include_dirs=[np.get_include()],
)
setuptools.setup(
    name="tf_pose",
    version="0.8.0",
    description="Deep Pose Estimation implemented using Tensorflow with Custom Architectures for fast inference.",
    long_description="This is a fork of the origninal https://github.com/ildoonet/tf-pose-estimation. Main change is "
                     "to restrict tensorflow 1.X. There are some incompatibility issue with tensorflow 2+ we haven't "
                     "sorted out. Our fork repository is https://github.com/tryagainconcepts/tf-pose-estimation",
    long_description_content_type="text/markdown",
    install_requires=REQUIRED_PACKAGES,
    dependency_links=DEPENDENCY_LINKS,
    url="https://github.com/ildoonet/tf-pose-estimation/",
    author="Ildoo Kim",
    author_email="ildoo@ildoo.net",
    license="Apache License 2.0",
    package_dir={"tf_pose_data": "models"},
    packages=["tf_pose_data"]
    + [
        pkg_name
        for pkg_name in setuptools.find_packages()  # main package
        if "tf_pose" in pkg_name
    ],
    ext_modules=[EXT],
    package_data={
        "tf_pose_data": ["graph/cmu/graph_opt.pb", "graph/mobilenet_thin/graph_opt.pb", "graph/cmu/download.sh"]
    },
    py_modules=["pafprocess"],
    zip_safe=False,
)
