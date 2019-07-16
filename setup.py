#!/usr/bin/env python
# coding=utf-8
import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="LensFinder",
    version="0.2",
    author="T.X. Mao",
    author_email="maotianxiang@bao.ac.cn",
    description="A ResNet based lens finder",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/POFK/LensFinder",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
