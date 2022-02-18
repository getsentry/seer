from setuptools import setup, find_packages

def get_requirements():
    with open(f"requirements.txt") as fp:
        return [x.strip() for x in fp.read().split("\n") if not x.startswith("#")]

setup(
    name="seer",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    install_requires=get_requirements(),
    tests_require=["pytest==7.0.1"]
)
