# BIOT
This project is based on BIR, a method for interpreting the dimensions of MDS embeddings/mappings (for more information about BIR, see https://github.com/AdrienBibal/BIR). BIOT is different from BIR with respect to 3 main points:
* BIOT can be applied to embeddings with more than two dimensions;
* Orthogonal transformations are used instead of rotations;
* Iterative optimization is used instead of performing an exhaustive search in a highly non-convex space.

This results in a method that converges automatically, that performs more transformations and, most importantly, that can now explain MDS embeddings/mappings with any number of dimensions (instead of only two, which is the case for BIR).

This code was written by Rebecca Marion and Adrien Bibal.

## Purpose of BIOT
The goal of BIOT is to find an explanation of MDS dimensions (no matter how many dimensions) by using as few external features as possible. By external features, we mean variables that can characterize the data at hand and that were not used to produce the MDS embedding/mapping.

## Pre-requisites
1) dataset.csv in the folder called Datasets. This file must contain the feature values used for explaining the embedding. Features must be quantitative (no categorical features).
Columns = features; rows = instances. The first row of the file contains the feature names.

2) embedding.csv in the folder called Datasets. This file must contain the embedding to explain. 
Columns = embedding dimensions; rows = instances. The first row of the file corresponds to the first instance.

## How do I run BIOT?
In order to run BIOT, execute all lines of BIOT.R. Inputs (the embedding X and the features Fe for explaining it) should be located in the folder Datasets. The results will be provided in the folder called Results.

If you want to run BIOT as a script (with Rscript), you can either use no arguments (in this case, dataset.csv and embedding.csv should be in the folder Datasets) or multiple arguments. If you use arguments, the two first arguments are mandatory: the first one is the path to the embedding file and the second one is the path to the dataset used to explain the embedding. The additional arguments must be in order, but are optional. The third argument is the path indicating where the results should be stored, the fourth is the start of the range of lambdas, the fifth is the end of the range and the sixth is the number of lambdas that should be evaluated.

Here is an example of how to run BIOT as a script, with the first two mandatory arguments and the first optinal one:<br>  Rscript BIOT.R a_path_to/my_embedding_file.csv another_path_to/my_dataset_file.csv an_output_path/output_file.RData
