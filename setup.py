import os
import pkg_resources
from setuptools import setup, find_packages


setup(
    name="T2IBenchmark",
    py_modules=["T2IBenchmark"],
    version="0.1.0",
    description="A benchmark for text2image generative models",
    author="Igor Pavlov, Artyom Ivanov, Stanislav Stafievskiy",
    url='https://github.com/boomb0om/text2image-benchmark',
    packages=find_packages(include=['T2IBenchmark', 'T2IBenchmark.*']),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ]
)