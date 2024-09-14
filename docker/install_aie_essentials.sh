#!/bin/bash

# Run Ryzen AI setup only if the file exists
if [ -f /IRON/npueval/docker/ryzen_ai-1.3.0ea1.tgz ]; then
    echo "Detected ryzen_ai essentials, setting up..."
    cd /IRON/npueval/docker && \
    tar -xzvf ryzen_ai-1.3.0ea1.tgz && \
    cd ryzen_ai-1.3.0 && \
    mv vitis_aie_essentials-*.whl /opt/ && \
    cd /opt && \
    mkdir vitis_aie_essentials && \
    unzip vitis_aie_essentials*.whl -d vitis_aie_essentials;
fi

# Setup license if exists
if [ -f /IRON/npueval/docker/Xilinx.lic ]; then
    cp /IRON/npueval/docker/Xilinx.lic /opt/ && \
    apt -y install locales iproute2 && \
    locale-gen en_US.UTF-8 && \
    echo 'export XILINXD_LICENSE_FILE=/opt/Xilinx.lic' >> /IRON/setup.sh && \
    echo 'export AIETOOLS_ROOT=/opt/vitis_aie_essentials' >> /IRON/setup.sh && \
    echo 'export AIE2_INCLUDE_DIR=${AIETOOLS_ROOT}/data/aie_ml/lib' >> /IRON/setup.sh && \
    echo 'export PATH=$PATH:${AIETOOLS_ROOT}/bin' >> /IRON/setup.sh && \
    echo 'export CHESSCCWRAP2_FLAGS="aie2 -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_ROOT}/include -D__AIENGINE__=2 -D__AIEARCH__=20"' >> /IRON/setup.sh && \
    echo 'export CHESSCCWRAP2P_FLAGS="aie2p -f -p me -P ${AIE2_INCLUDE_DIR} -I ${AIETOOLS_ROOT}/include -D__AIENGINE__=2 -D__AIEARCH__=20"' >> /IRON/setup.sh && \
    echo 'sudo ip link add vmnic0 type dummy || true' >> /IRON/setup.sh && \
    export MAC=$(grep -oP 'HOSTID=\K[^;]+' /opt/Xilinx.lic | head -n1 | sed 's/\(..\)/\1:/g; s/:$//') && \
    echo "sudo ip link set vmnic0 addr $MAC || true" >> /IRON/setup.sh;
fi
