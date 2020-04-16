# Contributing
Welcome! We are thrilled you are interested in contributing to Covasim. This document will help you get started.

## Notices
Contributions to this project are [released](https://help.github.com/articles/github-terms-of-service/#6-contributions-under-repository-license) to the public under the [project's open source license](LICENSE).


Please note that this project is released with a [Contributor Code of Conduct](https://github.com/InstituteforDiseaseModeling/covasim/blob/master/CODE_OF_CONDUCT.md). By participating in this project you agree to abide by its terms.

## Important Documents
Most documents you will need are inside the GitHub project. 

- [Issues](https://github.com/InstituteforDiseaseModeling/covasim/issues) for tracking bugs and requesting enhancements.
- [Pull Requests](https://github.com/InstituteforDiseaseModeling/covasim/pulls) for on going code and documentation contributions. 
- Readme file contain relevant information for the included components.
	- [Project Readme](https://github.com/InstituteforDiseaseModeling/covasim/blob/master/README.md) - Setup and project overview.
	- [Covasim Readme](https://github.com/InstituteforDiseaseModeling/covasim/blob/master/covasim/README.md) Simulation parameters.
	- [Web App Readme](https://github.com/InstituteforDiseaseModeling/covasim/blob/master/covasim/webapp/README.md) Running the web application.
	- [Docker Readme](https://github.com/InstituteforDiseaseModeling/covasim/blob/master/docker/README.md) Information regarding the docker setup.
	- [Tests Readme](https://github.com/InstituteforDiseaseModeling/covasim/tree/master/tests) Running tests locally.
- [Changelog](https://github.com/InstituteforDiseaseModeling/covasim/blob/master/CHANGELOG.md) 

## Contribution Types
This is a fast moving project with many opportunities to contribute across the project. We welcome the following types of contributions:

1. Issues:
	1. Bug reports. 
	1. Feature requests.
1. Pull Requests:
	1. Tests reproducing an issue. 
	1. Bug fixes.
	1. Code to resolve [open approved issues](https://github.com/InstituteforDiseaseModeling/covasim/issues?q=is%3Aopen+is%3Aissue+label%3Aapproved).   
	1. Documentation improvements.

All external communication about these contribution efforts is currently occurring on GitHub.

## Opening Issues
Before opening a new issue:

* **check your parameters [the README](https://github.com/InstituteforDiseaseModeling/covasim/blob/master/covasim/README.md)** to see if the behavior you observed might be expected and if configuration options are available to provide you with the desired behavior.
* **perform a cursory search** to see if there's [an existing issue](https://github.com/InstituteforDiseaseModeling/covasim/issues) covering your feedback. If there is one and the issue is still open, **add a :+1: reaction** on the issue to express interest in the issue being resolved. That will help the team gauge interest without the noise of comments which trigger notifications to all watchers. Comments should be used only if you have new and useful information to share.

When opening an issue for a feature request:

* **use a clear and descriptive title** for the issue to identify the problem.
* **include as many details as possible in the body**. Explain your use-case, the problems you're hitting and the solutions you'd like to see to address those problems.

When opening an issue for a bug report, explain the problem and include additional details to help maintainers reproduce the problem:

* **describe the exact steps which reproduce the problem** in as many details as possible. When listing steps, don't just say what you did, but explain how you did it.
* **provide specific examples to demonstrate the steps**. Include links to files or GitHub projects, or copy/pasteable snippets, which you use in those examples. If you're providing snippets in the issue, use Markdown code blocks.
* **describe the behavior you observed** after following the steps and point out what exactly is the problem with that behavior.
* **explain which behavior you expected to see instead** and why.

## Submitting a pull request

1. [Fork][fork] and clone the repository
1. Configure and install the dependencies as outline in the [README](https://github.com/InstituteforDiseaseModeling/covasim#detailed-installation-instructions)
1. Make sure the [tests pass on your machine](https://github.com/InstituteforDiseaseModeling/covasim/tree/master/tests#pytest).
1. Create a new branch: `git checkout -b my-branch-name`
1. Make your change, add tests, and make sure the tests still pass
1. Push to your fork and [submit a pull request][pr]
1. Pat your self on the back and wait for your pull request to be reviewed and merged.

Here are a few things you can do that will increase the likelihood of your pull request being accepted:

- Write tests.
- Keep your change as focused as possible. If there are multiple changes you would like to make that are not dependent upon each other, consider submitting them as separate pull requests.
- Write a [good commit message](http://tbaggery.com/2008/04/19/a-note-about-git-commit-messages.html).

## Resources

- [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
- [Using Pull Requests](https://help.github.com/articles/about-pull-requests/)
- [GitHub Help](https://help.github.com)