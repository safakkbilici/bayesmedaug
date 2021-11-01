from setuptools import setup, find_packages

setup(
    name = "bayesmedaug",
    packages=find_packages(),
    version = "0.1.1",
    license = "MIT",
    description = "Bayesian Optimization Library for Medical Image Segmentation.",
    author = "M. Şafak Bilici, Enes Sadi Uysal, Onur Boyar, Alara Hergün",
    author_email = "safakk.bilici.2112@gmail.com, enessadi@gmail.com, boyaronur@gmail.com, alarackck@gmail.com",
    url = "https://github.com/safakkbilici/bayesmedaug",
    download_url = TODO,
    keywords = [
        "Medical Image Segmentation",
        "Bayesian Optimization",
        "Data Augmentation"
    ],
    install_requires=[
        "bayesian-optimization",
        "torch>=1.6",
        "torchvision",
        "tqdm",
        "imgaug",
        "opencv-python"
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
    ],
)
