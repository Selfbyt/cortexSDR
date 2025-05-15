from setuptools import setup
from setuptools.extension import Extension
from Cython.Build import cythonize
import os

# Get the absolute path to the source directory
src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src'))

# Define the extension module
extensions = [
    Extension(
        "cortex_compression",
        sources=["cortex_compression.pyx"],
        include_dirs=[src_dir],
        language="c++",
        extra_compile_args=["-std=c++17"],
        extra_link_args=["-std=c++17"],
    )
]

setup(
    name="cortex-compression",
    version="0.1.0",
    description="High-performance AI model compression library",
    author="Your Name",
    author_email="your.email@example.com",
    ext_modules=cythonize(extensions),
    install_requires=[
        "numpy>=1.19.0",
        "torch>=1.7.0",
    ],
    python_requires=">=3.7",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
)
