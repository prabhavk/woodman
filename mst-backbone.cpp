#include <stdio.h>
#include <string>
#include "mstBackbone.h"
#include <iostream>
#include <experimental/filesystem>
// #include <boost/filesystem.hpp>
#include <sys/stat.h>
#include <boost/bind/bind.hpp>
using namespace boost::placeholders;
using namespace std;
namespace fs = std::experimental::filesystem;

int main(int argc, char **argv)
{	
	string path_to_alignment_file;
	string patch_name;
		// duplicate_sequences
		// highly similar sequnces (Hamming distance of 1; try upto 5)
		// incremental construction		
		// genotype-recombination network
		// Select subset of vertices based on phenotypes
		// Amino acid/Drug resistance/Phenotype network
		// Network layout
		// Dynamic network layout?		
	fs::path alignment_file_path_obj;
	fs::path prefix_path_obj;
    string prefix_for_output_files;
	string path_to_directory;
    int subtree_size;
 	struct stat buffer;
	// bool alignment_file_not_set = 0;
	bool flag_subtree_size = 0;
	bool flag_supertree = 0;
	bool flag_prefix = 0;
	bool flag_distance = 0;
	bool flag_root_supertree = 0;
	// bool localPhyloOnly = 0;
	bool flag_modelSelection = 0;
	bool flag_verbose = 0;
	// bool useChowLiu = 1;
	string modelForRooting = "UNREST";
	string root_supertree = "no";
	string arg_localPhyloOnly;
	string arg_modelSelection;
	string arg_chowliu;
	string string_verbose;
	string distance_measure = "logDet";
	string supertree_method = "mstbackbone";
	
	MSTBackbone * MSTBackboneObj;
    if (argc < 2) {        
		cerr << "Usage: " << argv[0] << " --seq alignment --subtree_size k --out filename " << endl;
        // cerr << "Usage: " << argv[0] << " --seq alignment.fas --subtree_size k --distance_measure LogDet --out filename --root_supertree no" << endl;
		cerr << endl;
        return (-1);
    } else {        
        // parse arguments            
        for (int i = 1; i < argc ; i++) {
        // path to multiple sequence alignment file
			if (strcmp(argv[i], "--seq") == 0) {				
                if (i < argc -1) {					
                    path_to_alignment_file = argv[++i];
					cout << "Alignment file name is " << path_to_alignment_file << endl;
					if (stat (path_to_alignment_file.c_str(), &buffer) != 0) { // check if input file exists
						cout << "Please check if the input filename is correct" << endl;
						exit (-1);
					}
					alignment_file_path_obj = path_to_alignment_file;
                }
        // prefix_to_output_files (e.g. prefix is "covid-19_size_42" and the alignment is in directory "/foo/bar/data/")		
            } else if (strcmp(argv[i], "--distance_measure") == 0) {
                if (i < argc -1) {
					distance_measure = argv[++i];
					flag_distance = 1;
					if (distance_measure == "logDet" || distance_measure == "Jukes-Cantor" ||distance_measure == "Hamming") {
						continue;
					} else {
						cout << "Enter one of the following distance measures: logDet, Jukes-Cantor, Hamming" << endl;
						exit (-1);
					}
                }
        // prefix_to_output_files (e.g. prefix is "covid-19_size_42" and the alignment is in directory "/foo/bar/data/")		
            }  else if (strcmp(argv[i], "--supertree_method") == 0) {
                if (i < argc -1) {
					supertree_method = argv[++i];
					flag_supertree = 1;
					if (supertree_method == "mstbackbone" || supertree_method == "pdrs") {
						continue;
					} else {
						cout << "Enter one of the following supertree methods: \n \t mstbackbone \n \t pdrs (place on degree-restricted subtree)" << endl;
						exit (-1);
					}
                }
        // prefix_to_output_files (e.g. prefix is "covid-19_size_42" and the alignment is in directory "/foo/bar/data/")		
            } else if (strcmp(argv[i], "--out") == 0) {
                if (i < argc -1) {
					flag_prefix = 1;
                    prefix_for_output_files = argv[++i];
					// prefix_for_output_files = alignment_file_path_obj.parent_path().string() + prefix_for_output_files;
					prefix_path_obj =  alignment_file_path_obj.parent_path();
					prefix_path_obj /= prefix_for_output_files;
					cout << "prefix for output files is " << prefix_for_output_files << endl;					
                }
        // size of subtree Vs
            } else if (strcmp(argv[i], "--verbose") == 0) {
                if (i < argc -1) {
			string_verbose = argv[++i];
			if (string_verbose == "True") {
				flag_verbose = 1;
			}			
                }        
            } else if (strcmp(argv[i], "--subtree_size") == 0) {
                if (i < argc -1) {
					flag_subtree_size = 1;
                    subtree_size = stoi(argv[++i]);
                }
			} else if (strcmp(argv[i], "--help") == 0) {
				cout << "Example for mst-constrained tree construction: " << argv[0] << " --seq alignment.fas --subtree_size k --distance_measure logDet --out prefix_for_output --root_supertree yes" << endl;
			} else if (strcmp(argv[i], "--root_supertree") == 0) {
				if (i < argc -1) {					
					root_supertree = argv[++i];
					if (root_supertree == "yes" || root_supertree == "y") {
						flag_root_supertree = 1;						
					} else if (root_supertree == "no" || root_supertree == "n") {
						flag_root_supertree = 0;
					} else {
						cout << "Enter one of the following options for root_supertree: yes, y, no, n" << endl;
						exit (-1);
					}					
                }
			}
        }

		if (!flag_subtree_size) {
			subtree_size = 10;
		}

		if (!flag_prefix) {
			prefix_path_obj =  alignment_file_path_obj.parent_path();
			prefix_path_obj /= "mstbackbone_output";
			// prefix_for_output_files = alignment_file_path_obj.parent_path().string() + "_mstbackbone";
		}		

		if (!flag_supertree) {
			supertree_method = "pdrs"; // place on degree-restricted subtree
		}		
		if (!flag_distance) {
			distance_measure = "LogDet"; // place on degree-restricted subtree
		}
		cout << "supertree method is " << supertree_method << endl;
		MSTBackboneObj = new MSTBackbone(path_to_alignment_file, subtree_size, prefix_path_obj.string(), distance_measure, flag_verbose, flag_root_supertree, supertree_method);		
		delete MSTBackboneObj;
		// MSTBackbone MSTBackboneObj(path_to_alignment_file, size_of_subtree, prefix_path_obj.string(),localPhyloOnly,modelSelection,modelForRooting,useChowLiu);
    }

	return 0;
} 
