import os
import runpy
from setuptools import setup, find_packages

# Load requirements from txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'covasim', 'version.py')
version = runpy.run_path(versionpath)['__version__']

CLASSIFIERS = [
    "Environment :: Console",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: GPLv3",
    "Operating System :: OS Independent",
    "Programming Language :: Python",
    "Topic :: Software Development :: Libraries :: Python Modules",
    "Development Status :: 1",
    "Programming Language :: Python :: 3.7",
]

setup(
    name="covasim",
    version=version,
    author="Cliff Kerr, Robyn Stuart, Romesh Abeysuriya, Dina Mistry, Lauren George, Mike Famulare, Daniel Klein",
    author_email="covid@idmod.org",
    description="Covid-19 agent-based model model",
    keywords=["Covid-19", "coronavirus", "cruise ship", "Diamond Princess", "Seattle", "agent-based model"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "matplotlib>=2.2.2",
        "numpy>=1.10.1",
        "scipy>=1.2.0",
        "sciris>=0.15.9",
        "scirisweb>=0.15.1",
        "pandas",
        "numba",
        "gunicorn",
        "plotly_express",
		# "parestlib>=0.3",
    ],
)
