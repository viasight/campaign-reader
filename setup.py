from setuptools import setup, find_packages

setup(
    name='campaign-reader',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=[
        'pandas>=1.3.0',  # Updated for newer DataFrame features
        'numpy>=1.21.0',  # Updated to match our array handling
        'opencv-python>=4.5.0',
        'opencv-python-headless>=4.5.0;platform_system!="Windows"',
    ],
    extras_require={
        'dev': [
            'pytest>=6.0.0',
            'pytest-mock>=3.6.0',
            'black',  # for code formatting
            'isort',  # for import sorting
            'flake8',  # for linting
        ],
        'test': [
            'pytest>=6.0.0',
            'pytest-mock>=3.6.0',
        ]
    },
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
        'Programming Language :: Python :: 3.8',  # Updated minimum Python version
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Private :: Do Not Upload',  # Indicates this is private package
    ],
    python_requires='>=3.8',  # Updated minimum Python version
)