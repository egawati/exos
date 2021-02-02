import codecs
from setuptools import setup, find_packages

entry_points = {
    'console_scripts': [
    ]
}


def _read(fname):
  with codecs.open(fname, encoding='utf-8') as f:
    return f.read()


setup(
    name='exos.streams',
    version=_read('version.txt').strip(),
    author='Egawati Panjei',
    author_email='egawati.panjei@gmail.com',
    description="Data Streams Simulator",
    long_description=(_read('README.rst') + '\n\n' + _read("CHANGES.rst")),
    license='Apache',
    keywords='Base',
    classifiers=[
        'Intended Audience :: Developers',
        'Natural Language :: English',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
    ],
    url="https://github.com/NextThought/nti.contenttools",
    zip_safe=True,
    packages=find_packages('src'),
    package_dir={'': 'src'},
    include_package_data=True,
    namespace_packages=['exos'],
    install_requires=[
        'setuptools',
        'numpy',
        'pandas',
        'scikit-multiflow',

    ],
    entry_points=entry_points
)
