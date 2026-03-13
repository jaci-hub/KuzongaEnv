from setuptools import setup, find_packages

setup(
    name="kuzongaenv",
    version="0.3.2",
    author="Jacinto Jeje Matamba Quimua",
    description="A custom Gymnasium-compatible environment for the Kuzonga",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.30.0",
        "numpy>=1.23"
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    url="https://github.com/jaci-hub/kuzongaenv",
    license="MIT",
    python_requires=">=3.11",
)
