import glob
import os
import shutil
import subprocess
import tempfile
from distutils.command.build import build
from os import path

from setuptools import find_packages, setup

here = path.abspath(path.dirname(__file__))


def get_requirements():
    with open(f"{here}/requirements.txt") as fp:
        return [x for x in fp.read().split("\n") if not x.startswith("#")]


def make_protos():
    protos = glob.glob(f"{here}/src/**/*.proto", recursive=True)
    with tempfile.TemporaryDirectory() as tmpd:
        for proto in protos:
            result = subprocess.run(
                [
                    "python",
                    "-m",
                    "grpc_tools.protoc",
                    f"-I{here}/src",
                    f"--python_out={tmpd}",
                    f"--mypy_out={tmpd}",
                    f"--grpc_python_out={tmpd}",
                    f"--mypy_grpc_out={tmpd}",
                    f"--python_adaptors_out={tmpd}",
                    proto,
                ]
            )
            assert result.returncode == 0, "protoc failed, check output above"

        for dir, _, files in os.walk(tmpd):
            if "__init__.py" in files:
                continue
            with open(f"{dir}/__init__.py", "w") as f:
                f.write("")

        for p in os.listdir(tmpd):
            p = f"{tmpd}/{p}"
            if not os.path.isdir(p):
                continue
            with open(f"{p}/py.typed", "w") as f:
                f.write("")

        shutil.rmtree(f"{here}/py")
        shutil.move(tmpd, f"{here}/py/")


class proto_build(build):
    def run(self):
        make_protos()
        super().run()


setup(
    name="sentry-protos",
    version="0.1.0",
    package_dir={"": f"{here}/py"},
    package_data={"": ["py.typed"]},
    packages=find_packages(where="py"),
    install_requires=get_requirements(),
    setup_requires=get_requirements(),
    cmdclass={"build": proto_build},
)
