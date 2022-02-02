# ibl-hdf5-creation

This script file data_creation.py takes in an eid (experiment id) and absolute directory 
path - both strings - and pulls and formats video, label, and neural data for decoding. It stores
this data in an HDF5 file broken up into groups (images, labels, neural) with each group populated
with datasets corresponding to each trial. Follows formatting given by the IBL Behavenet API:
https://behavenet.readthedocs.io/en/latest/source/data_structure.html#data-structure

To create data file: 
  1. Create/cd into directory for this repo
  2. Clone repo in directory
  3. cd into ibl-hdf5-creation folder
  4. Run "pip install requirements.txt"
  5. Run "pip install -e ."
  6. Run "python data_creation.py --save_dir [absolute_path] --eid [ONE experiment_id]
