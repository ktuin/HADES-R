from setuptools import setup, find_packages

setup(
    name="HadesR",
    version="1.0",
    description="One-master-event relative DGP event locator",
    author="Katinka Tuinstra",
    author_email="katinka.tuinstra@sed.ethz.ch",
    py_modules=["HadesR"],
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "matplotlib",
        "halo",
        "matplotlib",
        "e13tools",
        "datetime",
    ],
)
