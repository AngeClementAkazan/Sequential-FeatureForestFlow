from setuptools import setup, find_packages

with open("README", 'r') as f:
    long_description = f.read()

setup( 
    name='Forest FLow-Based Per Variable Sampling',
   version='1.1.0',
   description='This package implement a modified version of flow matching that is based on a per-variable sampling (generation) instead of generating the full data',
   license="MIT",
    packages=find_packages(),
   long_description=long_description,
   author='Ange-Clément Akazan',
   author_email='ange-clement.akazan@mila.quebec',
   url="https://github.com/AngeClementAkazan/Forest-FLow-Based-Variable-Sampling",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
   install_requires=['numpy','pandas','scikit-learn','pandas','xgboost>=2.0.0'] #external packages as dependencies
   )