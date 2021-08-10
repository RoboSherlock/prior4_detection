from setuptools import setup

setup(
   name='Prior4PE',
   version='0.1.1',
   author='Jesse Richter-Klug',
   author_email='jesse@uni-bremen.de',
   packages=['Prior4PE'],
   install_requires=[
       "tensorflow==2.4.0",
       "scipy==1.5.1",
       "sklearn",
   ],
)