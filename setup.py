from setuptools import find_packages, setup

setup(
    name='humanoidverse',
    version='0.0.1',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='HumanoidVerse: A Multi-Simulator Framework for HumanoidVerse: A Multi-Simulator Framework for ',
    url="https://github.com/LeCAR-Lab//HumanoidVerse",  # Update this with your actual repository URL
    python_requires=">=3.8",
    install_requires=[
        "hydra-core",
        "numpy",
        "rich",
        "ipdb",
        "matplotlib",
        "termcolor",
        "wandb",
        "plotly",
        "tqdm",
        "loguru",
        "meshcat",
        "pynput",
        "scipy",
        "tensorboard==2.12.0",
        "onnx",
        "onnxruntime",
        "opencv-python",
        "joblib",
        "easydict",
        "lxml",
        "numpy-stl",
        "open3d"
    ]
)