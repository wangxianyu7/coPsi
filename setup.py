from setuptools import find_packages, setup

dependencies=''
with open("requirements.txt","r") as f:
	dependencies = f.read().splitlines()

setup(
	name="PhotProt",
	author='Emil Knudstrup',
	author_email='emil@phys.au.dk',
	description='Calculate stellar rotation, inclination, and orientation.',
	packages=find_packages(where="src"),
	package_dir={"": "src"},
	classifiers = ["Programming Language :: Python :: 3"],
	install_requires = dependencies
)
