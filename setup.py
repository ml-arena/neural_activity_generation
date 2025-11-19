from setuptools import setup, find_packages

setup(
    name="neural_activity",
    version="0.2",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        'neural_activity': ['data/*.npz', 'data/*.npy'],  # Include neural data files
    },
    install_requires=[
        "numpy>=1.20.0",
        "scipy>=1.7.0",
    ],
)
