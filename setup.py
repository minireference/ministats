#!/usr/bin/env python

"""The setup script."""

from setuptools import setup, find_packages

with open('README.md') as readme_file:
    readme = readme_file.read()

with open('HISTORY.md') as history_file:
    history = history_file.read()


requirements = [
    "matplotlib>=3.7",
    "numpy>=1.24",
    "scipy>=1.6",
    "pandas>=2",
    "seaborn>=0.13.2",
    "statsmodels>=0.14.1",
]

test_requirements = ['pytest>=3', ]

setup(
    author="Ivan Savov",
    author_email='ivan@minireference.com',
    python_requires='>=3.6',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    description="Common statistical testing procedures used for STATS 101 topics. The code is intentionally simple to make it easy to forllow for beginners.",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    long_description_content_type="text/markdown",
    include_package_data=True,
    keywords='ministats',
    name='ministats',
    packages=find_packages(include=['ministats', 'ministats.*']),
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/minireference/ministats',
    version='0.3.2',
    zip_safe=False,
)
