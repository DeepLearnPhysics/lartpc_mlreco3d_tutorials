FROM deeplearnphysics/larcv2:ub18.04-cuda10.2-pytorch1.7.1-extra

RUN git clone https://github.com/Temigo/lartpc_mlreco3d.git /app/lartpc_mlreco3d
WORKDIR /app/lartpc_mlreco3d 
RUN git checkout develop

ARG NB_USER=jovyan
ARG NB_UID=1000
ENV USER ${NB_USER}
ENV NB_UID ${NB_UID}
ENV HOME /home/${NB_USER}

RUN adduser --disabled-password \
    --gecos "Default user" \
    --uid ${NB_UID} \
    ${NB_USER}
    
# Make sure the contents of our repo are in ${HOME}
WORKDIR /app
COPY . ${HOME}
USER root
RUN chown -R ${NB_UID} ${HOME}
USER ${NB_USER}

ENV PYTHONPATH ${PYTHONPATH}:${HOME}/lartpc_mlreco3d