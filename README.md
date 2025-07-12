# About mst-backbone

mst-backbone is a phylogeny inference program that uses minimum spanning trees to perform constrained search through tree-space. 

Here is the link to the preprint describing the performance of mst-backbone 

https://www.biorxiv.org/content/10.1101/2020.06.30.180315v1

# Software requirements

mst-backbone has been written in C++ (requires C++ 11) and has been compiled using g++ v11.3.0 with stdlib header enabled. The code is licensed under GPL 3.

# Installation


```console
git clone https://github.com/prabhavk/k2020.git 
cd k2020/
make mst-backbone

```

mst-backbone constructs phylogenetic trees by constraining the search through tree-space. Tree search is performed under the general Markov model using structural EM. 

 
UNRESTselector.py performs model selection by fitting non-stationary non-reversible non-homogeneous CT-HMM to phylogenetic trees. 

# Using mst-backbone

```console
mst-backbone --seq alignment.fas --constraint_size size_of_subtree --distance_measure_for_NJ logDet --out prefix_for_output --root_supertree no
```
Input file:

alignment.fas: a multiple sequence alignment file in fasta format

Parameters:

constraint_size: size constraint on the number of edges in the subgraphs induced by vertex sets $V_S$ and $V_O$ 

distance_measure_for_NJ: distance measure used for constructing NJ trees. Select among logDet, Jukes-Cantor and Hamming distance.

out: prefix used for naming output files. Default prefix is mstbackbone_output

root_supertree: Choose whether or not to root the supertree under the GMM via EM

Output files:

output_prefix.mst: the maximum parsimony spanning tree used for constraining search through tree-space 

output_prefix.unrooted_newick: the output tree in newick format

output_prefix.unrooted_edges: the output tree in edge list format

output_prefix.mstbackbone_log: contains all the messages that is printed to terminal

# Using UNRESTselector.py

The model selection procedure described in the paper can be reproduced using the following command

```console
python3 UNRESTselector.py --seq alignment.fas --out output_prefix --input_tree tree.newick
```

Input files:

alignment.fas: a multiple sequence alignment file in fasta format

tree.newick: a phylogenetic tree in newick format. The input tree is re-rooted such that BIC is minimized.

Parameters:

output_prefix: prefix used for naming output files. Default prefix is alignment_UNRESTselector.

Output files:

output_prefix.newick: the output tree in newick format 

output_prefix.edges: the output tree in edge list format 

output_prefix.params: the parameters of the selected model

output_prefix.log: log file containing BIC and time elapsed for performing model selection

# Data for reproducing results in the paper

All of the empirical data that are analyzed in the paper can be found in empirical_data.tar.gz. Tree identfiers for RAxML Grove database are stored in simulated_data.tar.gz.
