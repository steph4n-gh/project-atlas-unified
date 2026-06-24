from setuptools import setup, find_packages

setup(
    name="qan-transformers",
    packages=find_packages(exclude=["tests*", "benchmarks*", "scratch*"]),
)
