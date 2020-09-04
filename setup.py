from setuptools import setup

about = {}
with open("permutations_stats/__about__.py") as fp:
    exec(fp.read(), about)
version = about['__version__']

setup(
    name='permutations-stats',
    version=version,
    package_dirs={
        'permutations_stats': 'permutations_stats',
        'permutations_stats.tests': 'permutations_stats/tests'
    },
    python_requires='>3.6',
    install_requires=["numpy", "numba"],
    packages=['permutations_stats', 'permutations_stats.tests'],
    url='https://github.com/DarthTrevis/permutations-stats',
    license='All rights reserved (at the moment)',
    author='Florian Charlier',
    author_email='florian.charlier@bordet.be',
    description='Permutation-based statistical tests in Python'
)
