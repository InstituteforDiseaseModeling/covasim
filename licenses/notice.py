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
from pathlib import Path
from typing import Dict, List

from importlib_metadata import distribution
from typing.io import IO


NOTICE_FILE = "NOTICE"


project = Path(os.path.dirname(os.path.abspath(__file__))).parent.resolve()
excludes = ['pip', 'setuptools', 'wheel', 'pkg-resources', 'covasim']
encoding = "utf-8"


def get_pkg_name(path: Path):
    return re.sub(r'^(.*?)-\d+(\.\d+)+$', r'\1', path.stem).replace('_', '-')


def inflate(req: Dict[str, str], licenses: Dict[str, str]):
    pkg = req['name']
    dist = distribution(pkg)
    record = dict(package=pkg)
    record["version"] = f"{dist.metadata['Version']}"
    record["url"] = f"{dist.metadata['Home-page']}"
    record["license"] = f"{dist.metadata['License']}"
    record["license_text"] = licenses.get(pkg, "")
    return record


def get_licenses(lib=Path(sysconfig.get_path('platlib')), excludes=excludes):
    licenses = []
    licenses.extend(list(lib.rglob('*.dist-info/LICENSE')))
    licenses.extend(list(lib.rglob('*.dist-info/LICENCE')))
    licenses.extend(list(lib.rglob('*.dist-info/LICENSE.*')))
    licenses.extend(list(lib.rglob('*.dist-info/LICENCE.*')))

    packages = dict()

    for lic in licenses:
        pkg = get_pkg_name(lic.parent)
        if pkg not in packages and pkg not in excludes:
            packages[pkg] = lic.read_text(encoding=encoding)
    return packages


def get_requirements(licenses: Dict[str, str], excludes=excludes):
    reqs_json = subprocess.run(
            [
                os.path.join(sysconfig.get_path('scripts'), 'pip'),
                'list',
                '--local',
                '--format',
                'json'
            ],
            capture_output=True
    ).stdout.decode(encoding=encoding)
    reqs = json.loads(reqs_json, encoding=encoding)
    requirements = list(
            map(
                    lambda req: inflate(req, licenses),
                    filter(
                            lambda req: req['name'] not in excludes,
                            reqs
                    )
            )
    )
    return requirements


def as_csv(file, reqs: List[Dict[str, str]]):
    writer = csv.DictWriter(file, fieldnames=['package', 'version', 'url', 'license', 'license_text'], quoting=csv.QUOTE_ALL)
    writer.writeheader()
    writer.writerows(reqs)


def as_json(file, reqs: List[Dict[str, str]]):
    json.dump(reqs, file, indent=True)


def text_format(req: Dict[str, str]):
    record = []
    for key, value in req.items():
        if key != 'license_text':
            record.append(f"{key}: {value}")
        else:
            record.append("-" * 16)
            record.append(value)
    return record


def as_text(file: IO, reqs: List[Dict[str, str]]):
    records = list(map(text_format, reqs))
    nl = "\n"
    hr = "".join([nl, "=" * 80, nl])
    header = nl.join([
        "covasim",
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
    default_outfile = project.joinpath(NOTICE_FILE)
    parser = ArgumentParser(description="Generate license NOTICE file for 3rd party packages")
    parser.add_argument("-f", "--format", default="text", choices=['text', 'json', 'csv'])
    parser.add_argument("-o", "--outfile", default=default_outfile.as_posix())
    args = parser.parse_args()

    licenses = get_licenses()
    requirements = get_requirements(licenses)
    writer_fn = getattr(globals(), f"as_{args.format}".lower(), as_text)

    with open(args.outfile, "w", encoding="utf-8", newline="\n") as f:
        writer_fn(f, requirements)
        f.close()
