=======================
Contributing to Covasim
=======================

Thank you for your interest in improving the Covasim model! Please see the
guidelines below to provide feedback or contribute to Covasim code.  Note that
we make no representations that the code works as intended or that we will
provide support, address issues that are found, or accept pull requests.

Notices
=======

Contributions to this project are released to the public under the project's open source license.
See the primary README_ for more information.

.. _README: https://github.com/InstituteforDiseaseModeling/covasim/blob/master/README.rst

Note that this project is released with a contributor Code of Conduct. By participating in this project
you agree to abide by its terms.

Request new features or report bugs
===================================

If you notice unexpected behavior or a limitation in Covasim, follow the steps below before requesting a new feature or reporting a bug.

1.  First, review the Covasim documentation_ to see if there is already functionality that supports
    what you want to do.
2.  Search the existing issues_ to see if there is already one that contains your feedback. If there
    is, add a thumbs up reaction to convey your interest in the issue being addressed.

.. _documentation: https://institutefordiseasemodeling.github.io/covasim/index.html

.. _issues: https://github.com/InstituteforDiseaseModeling/covasim/issues


Open a feature request
----------------------

When opening an issue to request a new feature, do the following:

1.  Provide a clear and descriptive title for the issue.
2.  Include as many details as possible in the body. Fully explain your use case and,
    if possible, your proposed solution.

Report a bug
------------

When opening an issue to report a bug, explain the problem and include additional details to help us reproduce the problem:

1.  Describe the specific steps that led to the problem you encountered with as many details as possible.
    Don't just say what you did, but explain how you did it.
2.  Provide specific examples to demonstrate the steps, such as links to files or projects, code snippets,
    or screen shots. Please use Markdown styling for code snippets.
3.  Describe the behavior you observed after following the steps and point out exactly what the problem
    with that behavior is, including explaining what you expected to see instead and why.


Submit a pull request
=====================

To contribute directly to Covasim code, do the following:

1.  Fork and clone the Covasim repository.
2.  Install Covasim on your machine. See :doc:`installation` or the primary README_.
3.  Create a new branch::

        git checkout -b my-branch-name

4.  Make your code changes, including a descriptive commit message.
5.  Push to your fork and submit a pull request.

Although we make no guarantees that a submitted pull request will be merged, PRs
that meet the following criteria are more likely to be merged:

*   Up-to-date with master with no merge conflicts
*   Self-contained
*   Fix a demonstrable limitation of bug
*   Follow the current code style
*   If the PR introduces a new feature, it has complete `Google style docstrings`_ and comments,
    and a test demonstrating its functionality
*   Otherwise, sample code demonstrating old and new behavior (this can be in the PR comment on
    GitHub, not necessarily committed in the repo)

.. _Google style docstrings: https://www.sphinx-doc.org/en/master/usage/extensions/example_google.html

If you have additional questions or comments, contact covasim@idmod.org.
