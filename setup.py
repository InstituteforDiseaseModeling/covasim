import os
import runpy
from setuptools import setup, find_packages

# Load requirements from txt file
with open('requirements.txt') as requirements_file:
    # ensure EOLs are '\n' in case on windows and splits
    requirements = []
    for line in requirements_file.read().replace('\r\n', '\n').split('\n'):
        if line and line[0] != '#':
            requirements.append(line)


# Get version
cwd = os.path.abspath(os.path.dirname(__file__))
versionpath = os.path.join(cwd, 'covasim', 'cova_base', 'version.py')
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
    author="Cliff Kerr, Romesh Abeysuriya, Dina Mistry, Mike Famulare, Daniel Klein",
    author_email="ckerr@idmod.org",
    description="Covid-19 agent-based model model",
    keywords=["Covid-19", "coronavirus", "cruise ship", "Diamond Princess", "Seattle", "agent-based model"],
    platforms=["OS Independent"],
    classifiers=CLASSIFIERS,
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements
)
