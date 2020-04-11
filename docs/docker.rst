========================
Build and test in Docker
========================

The docker directory contains scripts and files related to building |Cov_s| into a Docker container. This is intended as an environment to allow you to build and test the Docker image locally using Docker only. To test the Kubernetes version, see the `.platform` directory.

Build the container
===================

 On Linux, you can build one of any of the following ways:

 * ``make build``
 * ``docker-compose build``

 On Windows, the recommended way is to run:

 * ``docker-compose build``



Run the latest build from CI
=============================

On Linux you can run:

.. code-block:: bash

    make pull-run


On Windows, you can run:

.. code-block:: bash

    docker-compose pull covasim
    docker-compose up -d


On both platforms, |Cov_s| will be available at http://localhost:8000