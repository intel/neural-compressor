## build `Neural Compressor(INC)` Containers:

### To build the the `Pip` based deployment container:
Please note that `INC_VER` must be set to a valid version published here:
https://pypi.org/project/neural-compressor/#history

```console
$ PYTHON=python3.8
$ INC_VER=1.12
$ IMAGE_NAME=neural-compressor
$ IMAGE_TAG=${INC_VER}
$ docker build --build-arg PYTHON=${PYTHON} --build-arg INC_VER=${INC_VER} -f Dockerfile -t ${IMAGE_NAME}:${IMAGE_TAG} .
```

###  To build the the `Pip` based development container:
Please note that `INC_BRANCH` must be a set to a valid branch name otherwise, Docker build fails.
If `${INC_BRANCH}-devel` does not meet Docker tagging requirements described here:
https://docs.docker.com/engine/reference/commandline/tag/
then please modify the tag so that the tagging requirement is met. For example replace `/` with `-`.

```console
$ PYTHON=python3.8
$ INC_BRANCH=v1.12
$ IMAGE_NAME=neural-compressor
$ IMAGE_TAG=${INC_BRANCH}-devel
$ docker build --build-arg PYTHON=${PYTHON} --build-arg INC_BRANCH=${INC_BRANCH} -f Dockerfile.devel -t ${IMAGE_NAME}:${IMAGE_TAG} .
```

### Check the Containers built:
```console
$ docker images | grep -i neural-compressor
neural-compressor                                                                           v1.12-devel                                               5c0dc1371312   5 minutes ago    2.76GB
neural-compressor                                                                           1.12                                                      303de7f7c38d   36 minutes ago   1.61GB
```
