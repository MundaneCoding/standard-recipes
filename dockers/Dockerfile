FROM pytorchlightning/pytorch_lightning:2.0.9.post0-py3.9-torch1.13-cuda11.8.0

ARG USERNAME=app
ARG GRPNAME=app
ARG USER_UID=1000
ARG USER_GID=1000

# create user
RUN groupadd --gid ${USER_GID} ${GRPNAME} \
    && useradd --uid ${USER_UID} -s /bin/bash --gid ${USER_GID} -m ${USERNAME} \
    && usermod -aG root ${USERNAME}

RUN pip install matplotlib numpy pandas h5py jupyter diffusers==0.12.1 torchvision==0.14.1

WORKDIR /workspace
RUN chown -R ${USERNAME}:${GRPNAME} /workspace