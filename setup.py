from setuptools import setup, find_packages

setup(
    name="aksara",
    version="2.0.0",
    author="Emylton Leunufna",
    description="AKSARA - Adaptive Knowledge & Semantic Architecture for Bahasa Representation & Autonomy",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov",
        ],
        "yaml": [
            "pyyaml>=6.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "aksara=aksara.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Natural Language :: Indonesian",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
