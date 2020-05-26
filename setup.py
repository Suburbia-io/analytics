from setuptools import find_packages, setup

setup(
    name="analytics",
    version="1.0.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=open("requirements.txt").readlines(),
    extras_require={"dev": open("requirements-dev.txt").readlines()},
    description="Analytics tools for CPG datasets",
    entry_points={},
    author="Ondrej",
    long_description_content_type="text/markdown",
    license="MIT",
)
