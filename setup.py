from setuptools import setup

setup(
   name='addnumbers',
   version='0.1.0',
   description='Adding 2 Numbers using Recurrent Networks',
   author='Vinit Jain',
   author_email='vinjain1011@gmail.com',
   packages=['addnumbers'],  #same as name
   install_requires=['numpy', 'torch'], #external packages as dependencies
)