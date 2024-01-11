# MFA-DTI
This repository contains the implementation and the associated datasets for the paper: *MFA-DTI : Drug-target interaction prediction based on multi-feature fusion adopted framework*

### Dependencies
* Python version 3.8.17
* The requirements.txt contains the full list of packages used

### Run and preprocess data
1. First, generate node embeddings:
`python ./VGNAE/main.py --dataset celegans`
2. Generate pre-training vectors for interaction features:
`python ./WalkPooling/main.py --data-name celegans`
3. Obtain chemical structure features:
`python ./HGP-SL/main.py --dataset celegans`
4. Obtain structure features for targets:
`python ./Fusion/protein_vector.py --dataset celegans`
5. Run final training and evaluation:
`python ./Fusion/main.py --dataset celegans`
6. The training data and generated vectors should be saved in the folder `/data`:
    - `data/amino` contains a list of amino acids and their structure vectors
    - `data/drug` features vectors of chemical structure
    - `data/dti` the DTI datasets
    - `data/dti_vectors` the node embedding vectors
    - `data/graph` the pre-trained interaction vectors
    - `data/protein` feature vectors of the target structures

