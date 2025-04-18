import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="micrograd",
    version="0.1.0",
    author="Andrej Karpathy",
    author_email="andrej.karpathy@gmail.com",
    description="A tiny autograd engine",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/karpathy/micrograd",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=['numpy'],
    python_requires='>=3.6',
)
