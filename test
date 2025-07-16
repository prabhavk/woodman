#!/usr/bin/env bash
echo "Running test on 156 H3N2 sequences" # add test for SARS-CoV-2 sequences
./mstbackbone  --seq H3N2_subsampled_5.fas --constraint_size 10 --out test_h3n2 
echo "Running test on 349 sequences from the experimental phylogeny data by Randall and colleagues (2016)"
./mstbackbone  --seq Randall2016_allSeqs.fas --constraint_size 20 --out test_Randall 