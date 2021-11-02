from setuptools import setup, find_packages
import sys

requirements = ['numpy', 'pandas', 'scipy']

setup_requirements = requirements + ['pytest-runner']
test_requirements = requirements + ['pytest']
install_requirements = requirements

setup(
    name='si',
    version='0.0.1',
    python_requires='>=3.7',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=install_requirements,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    author='',
    author_email='',
    description='Sistemas inteligentes',
    license='Apache License Version 2.0',
    keywords='',
    test_suite='tests',
)
