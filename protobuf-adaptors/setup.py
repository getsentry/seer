from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))


def get_requirements():
    with open(f"{here}/requirements.txt") as fp:
        return [x.strip() for x in fp.read().split("\n") if not x.startswith("#")]


setup(
    name="protobuf-adaptors",
    version="0.1.0",
    package_dir={"": f"{here}/src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements(),
    entry_points={
        "console_scripts": [
            "protoc-gen-python_adaptors = protobuf_adaptors.main:main",
        ]
    },
)
