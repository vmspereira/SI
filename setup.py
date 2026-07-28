from setuptools import setup, find_packages

requirements = ['numpy', 'pandas', 'scipy']

# cvxopt is only needed by the SVM, which solves its dual as a quadratic
# program. It is kept optional so the rest of the library installs without a
# solver toolchain: `pip install -e .[svm]` to get it.
extras = {'svm': ['cvxopt']}

setup_requirements = requirements + ['pytest-runner']
test_requirements = requirements + ['pytest']
install_requirements = requirements

setup(
    name='si',
    version='0.0.1',
    python_requires='>=3.9',
    package_dir={'': 'src'},
    packages=find_packages('src'),
    zip_safe=False,
    install_requires=install_requirements,
    extras_require=extras,
    setup_requires=setup_requirements,
    tests_require=test_requirements,
    author='Vítor Pereira',
    author_email='vmsapereira@gmail.com',
    description='Sistemas inteligentes',
    license='Attribution 4.0 International',
    keywords='',
    test_suite='tests',
)
