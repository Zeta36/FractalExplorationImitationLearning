from setuptools import find_packages, setup

setup(
    name="dqfd",
    description="dqfd agent + FractalAI.",
    version="0.0.1",
    license="MIT",
    author="Samu",
    author_email="guillem.db@fragile.tech",
    url="https://github.com/Zeta36/FractalExplorationImitationLearning",
    download_url="https://github.com/Zeta36/FractalExplorationImitationLearning",
    install_requires=[
        "plangym>=0.0.6",
        "fragile>=0.0.40",
        "numpy>=1.16.2",
        "gym>=0.10.9",
        "pillow-simd>=7.0.0.post3",
        "opencv-python>=4.2.0.32",
    ],
    packages=find_packages(),
    package_data={"": ["LICENSE", "README.md"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Console",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Topic :: Software Development :: Libraries",
    ],
)
