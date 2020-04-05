Covasim Docker
==============

This directory contains scripts and files related to building covasim into a Docker container. This is intended as an environment to allow you to build and test the docker image locally using docker only. To test the Kubernetes version, see the [platform](../.platform) folder.

# Building Container
 
 On Linux, you can build one of any of the following ways
 
 * `make build`
 * `docker-compose build`
 
 On Windows, the recommended way is to run 
 
 * `docker-compose build`



# Running the Latest Build from CI

On Linux you can run

```bash
make pull-run
```

On Windows, you can run

```bash
docker-compose pull covasim
docker-compose up -d
```

On both platforms, Covasim will be available at http://localhost:8000