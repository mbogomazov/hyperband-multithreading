from setuptools import setup, find_packages

VERSION = '0.0.1'
DESCRIPTION = 'Hyperband-multithreading'
LONG_DESCRIPTION = 'Hyperband library with multithreading and backuping state functions'

# Setting up
setup(
        name="hyperband_multithreading",
        version=VERSION,
        author="Mikhail Bogomazov",
        author_email="<youremail@email.com>",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[
            'hyperopt==0.2.5',
            'ipython==7.30.1',
            'keras==2.7.0',
            'numpy==1.19.2',
            'pandas==1.1.3',
            'scikit_learn==1.0.1',
            'xgboost==1.5.1'
        ],
        keywords=['python', 'data science', 'hyperopt', 'hyperband'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Education",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)