import os
from setuptools import setup, find_packages
from distutils.util import convert_path

main_ns = {}
ver_path = convert_path("poniard/_version.py")
with open(ver_path) as ver_file:
    exec(ver_file.read(), main_ns)

packages = find_packages(".", exclude=["*.test", "*.test.*"])

here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="poniard",
    version=main_ns["__version__"],
    description="Streamline scikit-learn model comparison",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Rafael Xavier",
    author_email="rxaviermontero@gmail.com",
    license="MIT",
    url="https://github.com/rxavier/poniard",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Intended Audience :: End Users/Desktop",
        "Topic :: Scientific/Engineering",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
    ],
    keywords=[
        "machine learning",
        "scikit-learn",
    ],
    install_requires=[
        "scikit-learn>=1.0.2",
        "xgboost>=1.5.0",
        "pandas>=1.3.5",
        "plotly",
        "tqdm",
    ],
    include_package_data=True,
    packages=packages,
    python_requires=">=3.7",
)
