from setuptools import find_packages, setup

setup(
    name="quanto",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/huggingface/quanto",
    author="David Corvoysier",
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
)
