#! /bin/bash
if ! [ -f ./beast.sif ]; then
  apptainer build beast.sif apptainer.def
  apptainer run --bind ./:/opt/pybeast beast.sif python3 main.py
else
  apptainer run --bind ./:/opt/pybeast beast.sif python3 main.py
fi
