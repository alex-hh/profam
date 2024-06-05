from setuptools import find_packages, setup

# Read the long description from the README
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="profam",
    version="0.1.0",
    description="Protein family language models",
    url="https://github.com/alex-hh/profam",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[
        "torch",
        "pandas",
        "transformers",
        "tokenizers",
        "datasets",
        "accelerate",
        "pre-commit",
        "lightning",
        "hydra-core",
    ],
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "train_command = src.train:main",
            "eval_command = src.eval:main",
        ]
    },
)
