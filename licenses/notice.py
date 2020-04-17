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

import importlib_metadata as im
from typing.io import IO


NOTICE_FILE = "NOTICE"

cwd = Path(os.path.dirname(os.path.abspath(__file__)))
project = cwd.parent.resolve()
project_name = 'covasim'
excludes = ['pip', 'setuptools', 'wheel', 'pkg-resources', project_name]
encoding = "utf-8"


class Record:
    name = ""
    version = ""
    url = ""
    license = ""
    license_text = ""

    def __init__(self, name="", version="", url="", license="", license_text=""):
        self.name = name
        self.version = version
        self.url = url
        self.license = license
        self.license_text = license_text


def get_pkg_name(path: Path):
    return re.sub(r'^(.*?)-\d+(\.\d+)+$', r'\1', path.stem).replace('_', '-')


def inflate(dist: im.Distribution, licenses: Dict[str, str]):
    record = Record()
    record.name = f"{dist.metadata['Name']}"
    record.version = dist.version
    record.url = f"{dist.metadata['Home-page']}"
    record.license = f"{dist.metadata['License']}"
    record.license_text = licenses.get(record.name, "")
    return record


def get_licenses(lib=Path(sysconfig.get_path('platlib')), excludes=excludes):
    license_files = []
    license_files.extend(list(lib.rglob('*.dist-info/LICEN[SC]E')))
    license_files.extend(list(lib.rglob('*.dist-info/LICEN[CS]E.*')))

    licenses = dict()

    for lic in license_files:
        pkg = get_pkg_name(lic.parent)
        if pkg not in licenses and pkg not in excludes:
            licenses[pkg] = lic.read_text(encoding=encoding)
    return licenses

def load_custom_requirements():
    entries =   cwd.glob('custom/*/package.json')
    reqs = []
    for req_path in cwd.glob('custom/*/package.json'):
        to_json = json.loads(req_path.read_text(encoding=encoding))
        rec = Record(**to_json)
        rec.license_text = req_path.parent.joinpath('LICENSE').read_text(encoding=encoding)
        reqs.append(rec)

    return reqs
def get_requirements(excludes=excludes):
    licenses = get_licenses(excludes=excludes)
    predicate = lambda dist: dist.metadata['Name'] not in excludes
    py_reqs = list(
            map(
                    lambda d: inflate(d, licenses),
                    filter(predicate, im.distributions())
            )
    )
    js_reqs_json = subprocess.run(
            ['node', cwd.joinpath('notice.js')],
            capture_output=True
    ).stdout.decode(encoding=encoding)
    js_reqs = list(
            map(
                    lambda r: Record(**r),
                    json.loads(js_reqs_json)
            )
    )

    custom_reqs = load_custom_requirements()
    reqs = [*py_reqs, *js_reqs, *custom_reqs]
    reqs.sort(key=lambda r: r.name)
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
    default_outfile = project.joinpath(NOTICE_FILE)
    parser = ArgumentParser(description="Generate license NOTICE file for 3rd party packages")
    parser.add_argument("-f", "--format", default="text", choices=['text', 'json', 'csv'])
    parser.add_argument("-o", "--outfile", default=default_outfile.as_posix())
    args = parser.parse_args()

    requirements = get_requirements(excludes)
    writer_fn = getattr(globals(), f"as_{args.format}".lower(), as_text)

    with open(args.outfile, "w", encoding="utf-8", newline="\n") as f:
        writer_fn(f, requirements)
        f.close()
