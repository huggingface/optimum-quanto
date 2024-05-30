from setuptools import find_packages, setup

setup(
    name="quanto",
    url="https://github.com/huggingface/quanto",
    author="David Corvoysier",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
