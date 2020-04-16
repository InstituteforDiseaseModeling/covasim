const fs = require('fs')
const path = require('path')
const process = require('process')

const repo = path.join(__dirname, '..')
const paths = {
    repo: repo,
    webapp: path.join(repo, 'covasim/webapp'),
    licenses: path.join(repo, 'licenses')
}

function handler(packages = {}) {
    return Object.values(packages)
                 .map((p) => {
                     return {
                         name: p.name,
                         version: p.version,
                         url: p.repository,
                         license: p.licenses,
                         license_text: p.licenseText
                     }
                 })
}

const checker = require('license-checker')
checker.init({
                 start: paths.webapp,
                 production: true,
                 excludePackages: 'covasim',
                 excludePrivatePackages: true,
                 direct: true,
                 customPath: path.join(paths.licenses, 'js-license-format.json')
             },
             (err, packages) => {
                 if (err) {
                     console.error(err)
                 } else {
                     let json = JSON.stringify(handler(packages))
                     console.log(json)
                 }
             })


