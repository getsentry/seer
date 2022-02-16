from setuptools import setup, find_packages

setup(
    name="seer",
    version="0.1.0",
    package_dir={"": "src"},
    packages=find_packages("src"),
    install_requires=[
        "prophet==1.0.1",
        "Cython==0.29.27",
        "Flask==2.0.3",
        "gunicorn==20.1.0",
        "numpy==1.22.2",
        "pandas==1.4.1",
        "requests==2.27.1",
        "scipy==1.8.0",
        "sentry-sdk==1.5.5",
        "tsmoothie==1.0.4"
    ],
    tests_require=["pytest==7.0.1"]
)
