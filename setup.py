from setuptools import setup, find_packages

with open("README", 'r') as f:
    long_description = f.read()

setup(
   name='Forest FLow-Based Per Variable Sampling',
   version='1.0',
   description='A useful module',
   license="MIT",
    packages=find_packages(),
   long_description=long_description,
   author='Ange-Clément Akazan',
   author_email='ange-clement.akazan@mila.quebec',
   url="https://github.com/AngeClementAkazan/Forest-FLow-Based-Variable-Sampling",
   packages=['Forest FLow-Based Per Variable Sampling'],  #same as name
   install_requires=['numpy','pandas','scikit-learn','pandas','xgboost>=2.0.0' ] #external packages as dependencies
 
)