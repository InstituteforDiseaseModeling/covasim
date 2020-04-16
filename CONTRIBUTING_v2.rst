Contributing
============

Welcome! We are thrilled you are interested in contributing to Covasim.
This document will help you get started.

Getting started
---------------

Contributions to this project are
`released <https://help.github.com/articles/github-terms-of-service/#6-contributions-under-repository-license>`__
to the public under the `project's open source license <LICENSE>`__.

Please note that this project is released with a `Contributor Code of
Conduct <CODE_OF_CONDUCT.md>`__. By participating in this project you
agree to abide by its terms.

What you should read first
--------------------------

Most documents you will need are inside the GitHub project.

-  `Issues <https://github.com/InstituteforDiseaseModeling/covasim/issues>`__
   for tracking bugs and requesting enhancements.
-  `Pull
   requests <https://github.com/InstituteforDiseaseModeling/covasim/pulls>`__
   for on going code and documentation contributions.
-  Readme file contain relevant information for the included components.

   -  `Project readme <README.md>`__: Setup and project overview.
   -  `Covasim readme <./covasim/README.md>`__: Simulation parameters.
   -  `Webapp readme <./covasim/webapp/README.md>`__: Running the web
      application.
   -  `Docker readme <./docker/README.md>`__: Information regarding the
      docker setup.
   -  `Tests readme <./tests/README.md>`__: Running tests locally.

-  `Changelog <CHANGELOG.md>`__

Contribution types
------------------

This is a fast moving project with many opportunities to contribute
across the project. We welcome the following types of contributions:

1. Issues:

   1. Bug reports.
   2. Feature requests.

2. Pull Requests:

   1. Tests reproducing an issue.
   2. Bug fixes.
   3. Code to resolve `open approved
      issues <https://github.com/InstituteforDiseaseModeling/covasim/issues?q=is%3Aopen+is%3Aissue+label%3Aapproved>`__.
   4. Documentation improvements.

All external communication about these contribution efforts is currently
occurring on GitHub.

Opening issues
--------------

Before opening a new issue:

-  **Check your parameters against `the
   readme <./covasim/README.md>`__** to see if the behavior you observed
   might be expected and if configuration options are available to
   provide you with the desired behavior.
-  **See if there is already `an existing
   issue <https://github.com/InstituteforDiseaseModeling/covasim/issues>`__**
   covering your feedback. If there is one and the issue is still open,
   **add a :+1: reaction** on the issue to express interest in the issue
   being resolved. That will help the team gauge interest without the
   noise of comments which trigger notifications to all watchers.
   Comments should be used only if you have new and useful information
   to share.

When opening an issue for a feature request:

-  **Use a clear and descriptive title** for the issue to identify the
   problem.
-  **Include as many details as possible in the body**. Explain your
   use-case, the problems you're hitting and the solutions you'd like to
   see to address those problems. There are feature requests and bug
   reports which should help.

When opening an issue for a bug report, explain the problem and include
additional details to help maintainers reproduce the problem. There's
more information on the bug report template, but in brief:

-  **Describe the exact steps which reproduce the problem** in as much
   detail as possible. When listing steps, don't just say what you did,
   but explain how you did it.
-  **Provide specific examples to demonstrate the steps**. Include links
   to files or GitHub projects, or copy/pasteable snippets, which you
   use in those examples. If you're providing snippets in the issue, use
   Markdown code blocks.
-  **Describe the behavior you observed** after following the steps and
   point out what exactly is the problem with that behavior.
-  **Explain which behavior you expected to see instead** and why.

Submitting a pull request
-------------------------

1. Fork and clone the repository
2. Configure and install the dependencies as outline in the
   `README <https://github.com/InstituteforDiseaseModeling/covasim#detailed-installation-instructions>`__
3. Make sure the `tests pass on your
   machine <https://github.com/InstituteforDiseaseModeling/covasim/tree/master/tests#pytest>`__.
4. Create a new branch: ``git checkout -b my-branch-name``
5. Make your change, add tests, and make sure the tests still pass
6. Push to your fork and submit a pull request
7. Pat your self on the back and wait for your pull request to be
   reviewed and merged.

We obviously can't make any promises about whether or not PRs will be
accepted, here are a few things you can do that will increase the
likelihood of your pull request being accepted:

1. Up-to-date with master with no merge conflicts
2. Self-contained (changes one and only one aspect of the model)
3. Fix a demonstrable limitation of bug
4. Follow the current code style
5. If the PR introduces a new feature: have complete docstrings (Google
   style) and in-line comments, and a test in the ``tests/devtests``
   folder demonstrating its functionality
6. If the PR fixes a bug: sample code demonstrating the old and new
   behavior (can be in the PR comment on GitHub, not necessarily
   committed in the repo).

Resources
---------

-  `LitCovid <https://www.ncbi.nlm.nih.gov/research/coronavirus/>`__
-  `MIDAS network <https://midasnetwork.us/covid-19/>`__
-  `How to contribute to open
   source <https://opensource.guide/how-to-contribute/>`__
-  `Using pull
   requests <https://help.github.com/articles/about-pull-requests/>`__
-  `GitHub help <https://help.github.com>`__

