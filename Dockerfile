ARG BASE_IMAGE=graphcore/pytorch

FROM ${BASE_IMAGE} as base

LABEL maintainer='Matej <matej.uhrin@cvut.cz>'

# Use the opt directory as our dev directory
WORKDIR /poptorch

ENV PYTHONUNBUFFERED TRUE

COPY requirements.dev .
COPY poptorch_energy .
COPY setup.py .

# Install python dependencies
RUN pip install --upgrade pip \
    && pip install --no-cache-dir wheel \
    && pip install --no-cache-dir -r requirements.dev

RUN pip install -e .

RUN pip list