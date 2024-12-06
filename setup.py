from setuptools import setup, find_packages

setup(
    name='campaign-reader',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.0.0',
        'numpy>=1.18.0',
    ],
    author='Zikomo Fields',
    author_email='zikomo@zfields.tech',
    description='A Python package for reading and analyzing campaign zip files',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/Zikomo/campaign-reader',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',
)