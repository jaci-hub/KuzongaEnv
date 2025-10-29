from setuptools import setup, find_packages

setup(
    name="divide21",
    version="0.1.0",
    author="Jacinto Jeje Matamba Quimua",
    description="A custom Gymnasium-compatible environment for the Divide21 game invented by Jacinto Jeje Matamba Quimua.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    install_requires=[
        "gymnasium>=0.29",
        "numpy",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
)
