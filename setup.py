import os
import re
import sys
import runpy
from setuptools import setup, find_packages

# Load requirements from txt file
with open('requirements.txt') as f:
    requirements = f.read().splitlines()

if 'nowebapp' in sys.argv:
    print('Performing standalone installation -- running as a web application will not work')
    sys.argv.remove('nowebapp')
    webapp_reqs = [
        'scirisweb',
        'gunicorn',
        'plotly_express'
    ]
    regex = re.compile('[\W]+.*\Z')  # compare requirements to just the package name (strip off version info)
    requirements = list(filter(lambda p: regex.sub('', p) not in webapp_reqs, requirements))

if 'full' in sys.argv:
    print('Performing full installation, including optional dependencies')
    raise NotImplementedError('full installation not functional yet (dependent packages still private)')
    sys.argv.remove('full')
    full_reqs = [
        'covid_healthsystems',
        'synthpops',
        'parestlib'
    ]
    requirements.extend(full_reqs)

print(f'reqs => {requirements}')

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
    install_requires=requirements
)
