DOCKER_IMAGE=tiyengar:base_xpu
# DOCKER_IMAGE=vllm_xpu:bkc_ww31
docker run --privileged -it --rm \
	-u root \
        --ipc=host --net=host --cap-add=ALL \
        --device /dev/dri:/dev/dri \
        -v /dev/dri/by-path:/dev/dri/by-path \
        -v /lib/modules:/lib/modules \
        -v /data/dataset/librispeech:/data \
        -v /data/model:/model \
        -v ${PWD}/logs:/logs \
	-v ${PWD}:/workspace \
        ${DOCKER_IMAGE} /bin/bash
