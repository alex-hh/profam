from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="profam",
    packages=["profam"],
    version="0.1.0",
    description="Protein family language models",
    license="MIT",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch",
        "pandas",
        "transformers",
        "tokenizers",
        "datasets",  # for tranception
        "accelerate",
        "pre-commit",
        # 'atom3d'
    ],
)
