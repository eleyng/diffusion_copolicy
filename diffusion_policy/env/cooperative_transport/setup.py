import setuptools

setuptools.setup(
    name="cooperative-transport",
    version="0.0.1",
    author="Eley Ng",
    author_email="eleyng@stanford.edu",
    description="Custom environment and model for training RL agents for cooperative transport.",
    url="https://github.com/eley-ng/cooperative-transport",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.7",
    license="MIT",
    install_requires=[
        "gym",
        "matplotlib",
        "numpy",
        "pillow",
        "pygame",
        "stable-baselines3",
        "tensorboard",
        "torch",
    ],
)