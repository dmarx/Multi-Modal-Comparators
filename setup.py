from setuptools import setup, find_packages
setup(
    name='mmc',
    version='0.2.0',
    install_requires=[
        "wheel",
        "loguru",
        "clip-anytorch",
        "napm",
    ],
    packages=find_packages(
        #where='src/mmc',
        where='src',
        include=['mmc*'],  # ["*"] by default
        #exclude=['mypackage.tests'],  # empty by default
    ),
    package_dir={'mmc': 'src/mmc'},
)
