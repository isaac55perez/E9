from setuptools import setup, find_packages

# Read the contents of README file
with open("README.md", encoding="utf-8") as f:
    long_description = f.read()

# Read requirements from requirements.txt
with open("requirements.txt", encoding="utf-8") as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="signal-convolution-exercise",
    version="1.0.0",
    description="A demonstration of signal convolution using sine waves and peak detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="",  # Add your name here
    author_email="",  # Add your email here
    url="https://github.com/yourusername/signal-convolution-exercise",  # Add your repository URL
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Signal Processing",
        "Topic :: Education :: Computer Aided Instruction (CAI)",
    ],
    python_requires=">=3.10",
    entry_points={
        "console_scripts": [
            "run-convolution=convolution:main",
        ],
    },
    keywords=[
        "signal processing",
        "convolution",
        "sine wave",
        "peak detection",
        "educational",
        "numpy",
        "matplotlib",
        "scipy"
    ],
)