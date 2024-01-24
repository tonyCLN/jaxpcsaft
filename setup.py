import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="tony_saft_jax",
    version="0.0.3",
    author="Antonio Cavalcante", #<<<
    author_email="tcavalcanteneto@gmail.com", #<<<
    description="A small pc-saft package jited to be faster", #<<<
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tonyCLN/tony_SAFT",  #<<<
    packages=setuptools.find_packages(
        where='.',
        include=['tony_saft*'],  # alternatively: `exclude=['additional*']`
        ),
    classifiers=[
        "Programming Language :: Python :: 3", 
        "Operating System :: OS Independent",
    ],
)
