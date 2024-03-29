from distutils.core import setup


setup(
    name="vlgp",
    version="2019.3.19",
    packages=["vlgp"],
    url="https://github.com/catniplab/vlgp",
    license="MIT",
    author="yuan",
    author_email="yuanz271@gmail.com",
    description="variational Latent Gaussian Process",
    python_requires=">=3.5.0",
    install_requires=["numpy", "scipy", "scikit-learn", "click"],
    classifiers=[
        # Trove classifiers
        # Full list: https://pypi.python.org/pypi?%3Aaction=list_classifiers
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.5",
    ],
)
