"""
Setup script for Industrial Safety Equipment Detection Model.
"""

from setuptools import setup, find_packages
import os

# Read the contents of your README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# Read requirements
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="industrial-safety-detection",
    version="1.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="YOLOv8-based object detection for industrial safety equipment",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/industrial-safety-detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0",
            "black>=22.0",
            "flake8>=5.0",
            "isort>=5.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-safety-detection=train_model:main",
            "detect-safety-equipment=inference:main",
            "evaluate-safety-model=evaluate_model:main",
            "analyze-safety-data=data_analysis:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.yaml", "*.yml", "*.txt", "*.md"],
    },
)
