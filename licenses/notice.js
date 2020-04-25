const fs = require('fs')
const path = require('path')
const process = require('process')

const repo = path.resolve(__dirname, '..')
const paths = {
    repo: repo,
    licenses: path.join(repo, 'licenses')
}

function handler(packages = {}) {
    return Object.values(packages)
                 .map((p) => {
                     return {
                         name: p.name,
                         version: p.version,
                         url: p.url || p.repository,
                         license: p.licenses,
                         license_text: p.licenseText
                     }
                 })
}

const checker = require('license-checker')
checker.init(
    {
        start: paths.licenses,
        production: true,
        excludePackages: 'covasim',
        direct: 0,
        customPath: path.join(paths.licenses, 'custom/license-format.json')
    },
    (err, packages) => {
        if (err) {
            console.error(err)
        } else {
            const transformed = handler(packages)
            let json = JSON.stringify(transformed)
            console.log(json)
        }
    }
)


