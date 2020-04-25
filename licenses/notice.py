'''
Scraper to auto-generate the NOTICE file.
'''

import csv
import json
import os
import re
import subprocess
import sysconfig
from argparse import ArgumentParser
from pathlib import Path, PurePath
from typing import Dict, List, Union
from packaging.requirements import REQUIREMENT
import importlib_metadata as im
from typing.io import IO
from pprint import pprint

NOTICE_FILE = "NOTICE"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
project = cwd.parent.resolve()
project_name = 'covasim'
encoding = "utf-8"
config = None

class Config:
    encoding = encoding
    format = 'text'
    excludes = ['pip', 'setuptools', 'wheel', 'pkg-resources', project_name]
    includes = list(map(lambda r: REQUIREMENT.parseString(r)[0],
                        im.distribution(project_name).requires))
    verbose = False
    graph = False
    outfile = cwd.joinpath(NOTICE_FILE).as_posix()

    def __init__(self, **kwargs):
        for key,val in kwargs.items():
            setattr(self, key, val)
        if self.graph:
            self.includes.clear()

    def exclude_package(self, name:str):
        return self.excludes is not None and name in self.excludes

    def include_package(self, name:str):
        return not self.exclude_package(name) \
               and self.includes is not None and name in self.includes
    def keep_dist(self, dist: im.Distribution):
        return self.include_package(dist.metadata['Name'])

class Record:
    name = ""
    version = ""
    url = ""
    license = ""
    license_text = ""

    def __init__(self, name="", version="", url="", license="", license_text="", **kwargs):
        self.name = name
        self.version = version
        self.url = url
        self.license = license
        self.license_text = license_text

    @staticmethod
    def get_license(dist_path:Union[ Path,str, PurePath]):
        license_text = ""

        dist_dir = Path(dist_path)
        license_files = []
        license_files.extend(list(dist_dir.rglob('LICEN[SC]E')))
        if len(license_files) == 0:
            license_files.extend(list(dist_dir.rglob('LICEN[SC]E.*')))

        if len(license_files) > 0:
            license_text = license_files[0].read_text(encoding=encoding)

        return license_text

    @staticmethod
    def from_distribution(dist: im.Distribution):
        record = Record()
        record.name = f"{dist.metadata['Name']}"
        record.version = dist.version
        record.url = f"{dist.metadata['Home-page']}"
        record.license = f"{dist.metadata['License']}"
        record.license_text = Record.get_license(dist.locate_file('.'))

        return record

    @staticmethod
    def from_name(name:str):
        try:
            dist = im.distribution(name)
            return Record.from_distribution(dist)
        except im.PackageNotFoundError as e:
            print(f"No package named '{name}' found.")


def get_pkg_name(path: Path):
    return re.sub(r'^(.*?)-\d+(\.\d+)+$', r'\1', path.stem).replace('_', '-')



def get_licenses(lib=Path(sysconfig.get_path('platlib'))):
    license_files = []
    license_files.extend(list(lib.rglob('*.dist-info/LICEN[SC]E')))
    license_files.extend(list(lib.rglob('*.dist-info/LICEN[CS]E.*')))

    licenses = dict()

    for lic in license_files:
        pkg = get_pkg_name(lic.parent)
        if pkg not in licenses and pkg not in excludes:
            licenses[pkg] = lic.read_text(encoding=encoding)
    return licenses
def get_js_requirements(config:Config):
    js_reqs_json = subprocess.run(
            ['node', cwd.joinpath('notice.js')],
            capture_output=True
    ).stdout.decode(encoding=encoding)

    return list(
            map(
                    lambda r: Record(**r),
                    json.loads(js_reqs_json)
            )
    )

def get_requirements(config:Config):
    py_reqs = []
    if config.graph:
        py_reqs = list(
                map(
                        Record.from_distribution,
                        filter(config.keep_dist, im.distributions())
                )
        )
    else:
        py_reqs = list(
                map(
                        Record.from_name,
                        config.includes
                )
        )
    py_reqs = list(filter(lambda r: r is not None, py_reqs))
    js_reqs = get_js_requirements(config)

    reqs = [*py_reqs, *js_reqs]
    reqs.sort(key=lambda r: r.name)
    if config.verbose:
        for req in reqs:
            pprint(req.__dict__)
    return reqs


def as_csv(file, reqs: List[Record]):
    fieldnames = list(vars(Record()).keys())
    writer = csv.DictWriter(file, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(reqs)


def as_json(file, reqs: List[Record]):
    json.dump(reqs, file, indent=True)


def text_format(req: Record):
    record = []
    for key, value in vars(req).items():
        if key != 'license_text':
            record.append(f"{key}: {value}")
        else:
            record.append("-" * 16)
            record.append(value)
    return record


def as_text(file: IO, reqs: List[Record]):
    records = list(map(text_format, reqs))
    nl = "\n"
    hr = "".join([nl, "=" * 80, nl])
    header = nl.join([
        project_name,
        "THIRD - PARTY SOFTWARE NOTICES AND INFORMATION",
        "This project incorporates components from the projects listed below.",
        hr,
        nl
    ])

    footer = nl.join([hr, nl])
    file.write(header)
    for record in records:
        file.write(nl.join(record))
        file.write(footer)
        file.flush()


if __name__ == '__main__':
    parser = ArgumentParser(description="Generate license NOTICE file for 3rd party packages")
    parser.add_argument("-f", "--format", default="text", choices=['text', 'json', 'csv'])
    parser.add_argument("-o", "--outfile", default=Config.outfile, help=f"Path to output file. Default: {Config.outfile}")
    parser.add_argument("-v", "--verbose", action="store_true", help="Output more verbose logging.")
    parser.add_argument("-g", "--graph", action="store_true", help="Include transitive dependencies")
    args = parser.parse_args()

    config = Config(**args.__dict__)
    if args.verbose:
        pprint(args)
        pprint(config.__dict__)
    requirements = get_requirements(config)
    writer_fn = getattr(globals(), f"as_{args.format}".lower(), as_text)

    with open(config.outfile, "w", encoding="utf-8", newline="\n") as f:
        writer_fn(f, requirements)
        f.close()
