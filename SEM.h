#ifndef SEM_H
#define SEM_H

#include <random>
#include <chrono>
#include <eigen3/Eigen/Dense>
#include <eigen3/unsupported/Eigen/MatrixFunctions>
#include <boost/algorithm/string.hpp>
#include <fstream>
#include <math.h>
#include <iomanip>
#include "MST.h"
//#include <boost/math/tools/minima.hpp>
using namespace Eigen;

class SEM_vertex {	
public:
	int degree = 0;
	int timesVisited = 0;
	bool observed = 0;	
    vector <unsigned char> compressedSequence;
	vector <string> dupl_seq_names;
	int id = -42;
	int global_id = -42;
	string newickLabel = "";
	string name = "";
	float logScalingFactors = 0;
	float vertexLogLikelihood = 0;
	float sumOfEdgeLogLikelihoods = 0;
	int rateCategory = 0;
	int GCContent = 0;	
	vector <SEM_vertex *> neighbors;
	vector <SEM_vertex *> children;
	SEM_vertex * parent = this;
	void AddNeighbor(SEM_vertex * v_ptr);
	void RemoveNeighbor(SEM_vertex * v_ptr);
	void AddParent(SEM_vertex * v_ptr);
	void RemoveParent();
	void AddChild(SEM_vertex * v_ptr);
	void RemoveChild(SEM_vertex * v_ptr);
	void SetVertexLogLikelihood(float vertexLogLikelihoodToSet);	
	int inDegree = 0;
	int outDegree = 0;
	Matrix4f transitionMatrix;
	Matrix4f transitionMatrix_stored;	
	array <float, 4> rootProbability;
	array <float, 4> posteriorProbability;	
	SEM_vertex (int idToAdd, vector <unsigned char> compressedSequenceToAdd) {
		this->id = idToAdd;
		this->compressedSequence = compressedSequenceToAdd;
		this->transitionMatrix = ArrayXXf::Zero(4,4);
		this->transitionMatrix_stored = ArrayXXf::Zero(4,4);
		for (int dna = 0; dna < 4; dna ++) {
			this->transitionMatrix(dna,dna) = 1.0;
			this->transitionMatrix_stored(dna,dna) = 1.0;
		}
		for (int i = 0; i < 4; i ++) {
			this->rootProbability[i] = 0;
			this->posteriorProbability[i] = 0;
		}
	}	
	~SEM_vertex () {
		this->neighbors.clear();
	}
};

void SEM_vertex::SetVertexLogLikelihood(float vertexLogLikelihoodToSet) {
	this->vertexLogLikelihood = vertexLogLikelihoodToSet;
}

void SEM_vertex::AddParent(SEM_vertex * v) {
	this->parent = v;
	this->inDegree += 1;
}

void SEM_vertex::RemoveParent() {
	this->parent = this;
	this->inDegree -=1;
}

void SEM_vertex::AddChild(SEM_vertex * v) {
	this->children.push_back(v);
	this->outDegree += 1;
}

void SEM_vertex::RemoveChild(SEM_vertex * v) {
	int ind = find(this->children.begin(),this->children.end(),v) - this->children.begin();
	this->children.erase(this->children.begin()+ind);
	this->outDegree -=1;
}

void SEM_vertex::AddNeighbor(SEM_vertex * v) {
	this->degree += 1;
	this->neighbors.push_back(v);
}

void SEM_vertex::RemoveNeighbor(SEM_vertex * v) {
	this->degree -= 1;
	int ind = find(this->neighbors.begin(),this->neighbors.end(),v) - this->neighbors.begin();
	this->neighbors.erase(this->neighbors.begin()+ind);
}

class clique {
	public:	
	map <clique *, float> logScalingFactorForMessages;
	float logScalingFactorForClique;
	map <clique *, std::array <float, 4>> messagesFromNeighbors;
    vector <unsigned char> compressedSequence;
	string name;
	int id;
	int inDegree = 0;
	int outDegree = 0;
	int timesVisited = 0;
	clique * parent = this;
	vector <clique *> children;
	void AddParent(clique * C);
	void AddChild(clique * C);
	void ComputeBelief();
	SEM_vertex * x;
	SEM_vertex * y;
	std::array <float, 4> MarginalizeOverVariable(SEM_vertex * v);
//	Matrix4f DivideBeliefByMessageMarginalizedOverVariable(SEM_vertex * v);
	Matrix4f DivideBeliefBySepSetMarginal(clique * C);
	// Clique is defined over the vertex pair (X,Y)
	// No of variables is always 2 for bifurcating tree-structured DAGs
	
	Matrix4f initialPotential;	
	Matrix4f belief;
	// P(X,Y)
	
	void SetInitialPotentialAndBelief(int site);
	
	// If the clique contains an observed variable then initializing
	// the potential is the same as restricting the corresponding
	// CPD to row corresponding to observed variable
	void AddNeighbor(clique * C);
	clique (SEM_vertex * x, SEM_vertex * y) {
		this->x = x;
		this->y = y;
		this->name = to_string(x->id) + "-" + to_string(y->id);
		this->logScalingFactorForClique = 0;
	}
	
	~clique () {
		
	}
};

Matrix4f clique::DivideBeliefBySepSetMarginal(clique * C) {
	std::array <float, 4> message;
	SEM_vertex * variableNotInCommon;
	SEM_vertex * variableInCommon;
	if (this->x == C->x or this->x == C->y) {
		variableNotInCommon = this->y;
		variableInCommon = this->x;
	} else {
		variableNotInCommon = this->x;
		variableInCommon = this->y;
	}
	message = this->MarginalizeOverVariable(variableNotInCommon);
	Matrix4f factorAfterDivision;
	factorAfterDivision = this->belief;
	// 0/0 is defined as 0
	if (this->y == variableInCommon) {		
		// Perform element-wise division for each row of factor
		for (int dna_x = 0; dna_x < 4; dna_x ++) {
			for (int dna_y = 0; dna_y < 4; dna_y ++) {
				if (message[dna_y] == 0) {
					if (factorAfterDivision(dna_x,dna_y) == 0) {
						factorAfterDivision(dna_x,dna_y) = 0;
					} else {
						cout << "Dividing non-zero number by zero" << endl;
						exit (-1);
					}
				} else {
					factorAfterDivision(dna_x,dna_y) /= message[dna_y];
				}
			}
		}
	} else if (this->x == variableInCommon) {
		// Perform element-wise division for each column of factor
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			for (int dna_x = 0; dna_x < 4; dna_x ++) {
				if (message[dna_x] == 0) {
					if (factorAfterDivision(dna_x,dna_y) == 0) {
						factorAfterDivision(dna_x,dna_y) = 0;
					} else {
						cout << "Dividing non-zero number by zero" << endl;
						exit (-1);
					}					
				} else {
					factorAfterDivision(dna_x,dna_y) /= message[dna_x];
				}
			}
		}
	} else {
		cout << "Error in factor division operation" << endl;
		exit (-1);
	}
	return (factorAfterDivision);
}

std::array <float, 4> clique::MarginalizeOverVariable(SEM_vertex * v) {
	std::array <float, 4> message;	
	if (this->x == v) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			message[dna_y] = 0;
			for (int dna_x = 0; dna_x < 4; dna_x ++) {
				message[dna_y] += this->belief(dna_x,dna_y);
			}
		}
	} else if (this->y == v) {
		for (int dna_x = 0; dna_x < 4; dna_x ++) {
			message[dna_x] = 0;
			for (int dna_y = 0; dna_y < 4; dna_y ++) {
				message[dna_x] += this->belief(dna_x,dna_y);
			}
		}
	} else {
		cout << "Check marginalization over variable" << endl;
	}
	return (message);
}

void clique::ComputeBelief() {
	Matrix4f factor = this->initialPotential;
	vector <clique *> neighbors = this->children;
	std::array <float, 4> messageFromNeighbor;
	bool debug = 0;	
	if (debug) {
		cout << "Computing belief for clique " << this->name << endl;
		cout << "Factor before multiplying with belief" << endl;
		cout << factor << endl;		
	}
	if (this->parent != this) {
		neighbors.push_back(this->parent);
	}
	for (clique * C_neighbor : neighbors) {		
		this->logScalingFactorForClique += this->logScalingFactorForMessages[C_neighbor];
		messageFromNeighbor = this->messagesFromNeighbors[C_neighbor];
		if (this->y == C_neighbor->x or this->y == C_neighbor->y) {
//		factor_row_i = factor_row_i (dot) message
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
				for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
					factor(dna_x, dna_y) *= messageFromNeighbor[dna_y];					
				}
			}
			if (debug) {
				cout << "Performing row-wise multiplication" << endl;
			}			
		} else if (this->x == C_neighbor->x or this->x == C_neighbor->y) {
//		factor_col_i = factor_col_i (dot) message
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
					factor(dna_x, dna_y) *= messageFromNeighbor[dna_x];
				}
			}
			if (debug) {
				cout << "Performing column-wise multiplication" << endl;
			}			
		} else {
			cout << "Check product step" << endl;
			exit (-1);
		}
	}	
	float scalingFactor = 0;
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			scalingFactor += factor(dna_x, dna_y);
		}
	}	
	
	if (scalingFactor <= 0 || debug) {		
		cout << "Initial potential for " << this->name << " is " << endl << initialPotential << endl;
		for (clique * C_neighbor : neighbors) {		
			cout << "Message from neighbor " << C_neighbor->name << " is " << endl;
			for (int i = 0; i < 4; i ++) {
				cout << this->messagesFromNeighbors[C_neighbor][i] << "\t";
			}
			cout << endl;
		}
		cout << "========================" << endl;
	}
	// cout << "scalingFactor is " << scalingFactor << endl;
	assert(scalingFactor > 0);
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			this->belief(dna_x,dna_y) = factor(dna_x, dna_y)/scalingFactor;
		}
	}
	this->logScalingFactorForClique += log(scalingFactor);
}

void clique::AddParent(clique * C) {
	this->parent = C;
	this->inDegree += 1; 
}

void clique::AddChild(clique * C) {
	this->children.push_back(C);
	this->outDegree += 1;
}

void clique::SetInitialPotentialAndBelief(int site) {	
	// Initialize psi
	// V = (X,Y) X->Y (wlog), X is always an unobserved vertex
	int matchingCase;
	// Case 1. Y is an observed vertex
	// Product factor psi = P(Y|X) restricted to observed value xe of X
	// psi (y|x) is set to 0 if x != xe
	if (y->observed) {
		matchingCase = 1;
		this->initialPotential = y->transitionMatrix;
		int dna_y = y->compressedSequence[site];
		for (int dna_p = 0; dna_p < 4; dna_p++) {
			for (int dna_c = 0; dna_c < 4; dna_c++) {
				if (dna_c != dna_y) {
					this->initialPotential(dna_p,dna_c) *= 0;
				} else {
					this->initialPotential(dna_p,dna_c) *= 1;
				}
			}
		}		
	}
	
	// Case 2. X and Y are hidden and X is not the root
	// psi = P(Y|X)
	if (!y->observed) {
		matchingCase = 2;
		this->initialPotential = y->transitionMatrix;		
	}	
	
	// Case 3. X and Y are hidden and X is the root and "this" is not the root clique
	// psi = P(Y|X)
	if (!y->observed and (x->parent == x) and (this->parent != this)) {
		matchingCase = 3;
		this->initialPotential = y->transitionMatrix;
	}
	
	// Case 4. X and Y are hidden and X is the root and "this" is the root clique
	// psi = P(X) * P(Y|X) 
	if (!y->observed and (x->parent == x) and (this->parent == this)) {	
		matchingCase = 4;
		this->initialPotential = y->transitionMatrix;
		for (int dna_p = 0; dna_p < 4; dna_p++) {
			for (int dna_c = 0; dna_c < 4; dna_c++) {
				this->initialPotential(dna_p, dna_c) *= x->rootProbability[dna_c];
			}
		}
	}
	float maxValue = 0;
	for (int i = 0; i < 4; i ++) {
		for (int j = 0; j < 4; j ++) {
			if (this->initialPotential(i, j) > maxValue) {
				maxValue = this->initialPotential(i, j);
			}
		}
	}	
	if (maxValue < 0.001) {
		cout << "Matching case is " << matchingCase << endl;
		cout << "is x observed? " << x->observed << endl;
		cout << x->name << endl;
		cout << "is y observed? " << y->observed << endl;
		cout << y->name << endl;
		cout << "character at y is " << y->compressedSequence[site] << endl;
		cout << "char pos is " << site << endl;
		cout << this->name << endl;
		cout << this->x->name << "\t" << this->y->name << endl;
		cout << "Transition matrix is " << endl;
		cout << y->transitionMatrix << endl;
		for (int i = 0; i < 4; i ++) {
			cout << this->x->rootProbability[i] << endl;
		}
	}
	assert(maxValue > 0.001);
	this->belief = this->initialPotential;
	this->logScalingFactorForClique = 0;
	this->logScalingFactorForMessages.clear();
	this->messagesFromNeighbors.clear();
}

class cliqueTree {
public:
	vector < pair <clique *, clique *> > edgesForPreOrderTreeTraversal;
	vector < pair <clique *, clique *> > edgesForPostOrderTreeTraversal;
	vector < pair <clique *, clique *> > cliquePairsSortedWrtLengthOfShortestPath;
	map < pair <SEM_vertex *, SEM_vertex *>, Matrix4f> marginalizedProbabilitiesForVariablePair;
	map < pair <clique *, clique *>, pair <SEM_vertex *, SEM_vertex *>> cliquePairToVariablePair;
	int site;
	clique * root;
	bool rootSet;
	vector <clique *> leaves;
	vector <clique *> cliques;
	void CalibrateTree();
	void ComputeMarginalProbabilitesForEachEdge();
	void ComputeMarginalProbabilitesForEachVariablePair();
	void ComputePosteriorProbabilitesForVariable();
	void ConstructSortedListOfAllCliquePairs();
	clique * GetLCA (clique * C_1, clique * C_2);
	int GetDistance(clique * C_1, clique * C_2);
	int GetDistanceToAncestor(clique * C_d, clique * C_a);
	void SetLeaves();
	void SetRoot();
	void AddEdge(clique * C_1, clique * C_2);
	void SendMessage(clique * C_1, clique * C_2);
	void AddClique(clique * C);
	void SetSite(int site);
	void InitializePotentialAndBeliefs();
	void SetEdgesForTreeTraversalOperations();
	void WriteCliqueTreeAndPathLengthForCliquePairs(string fileName);
	Matrix4f GetP_XZ(SEM_vertex * X, SEM_vertex * Y, SEM_vertex * Z);
	SEM_vertex * GetCommonVariable(clique * Ci, clique * Cj);
	tuple <SEM_vertex *,SEM_vertex *,SEM_vertex *> GetXYZ(clique * Ci, clique * Cj);
	cliqueTree () {
		rootSet = 0;
	}
	~cliqueTree () {
		for (clique * C: this->cliques) {
			delete C;
		}
		this->cliques.clear();
		this->leaves.clear();
	}
};


tuple <SEM_vertex *,SEM_vertex *,SEM_vertex *> cliqueTree::GetXYZ(clique * Ci, clique * Cj) {
	SEM_vertex * X; SEM_vertex * Y; SEM_vertex * Z;
	SEM_vertex * Y_temp;
	clique * Cl;
	pair <clique *, clique *> cliquePairToCheck;
	if (Ci->parent == Cj or Cj->parent == Ci) {
		// Case 1: Ci and Cj are neighbors
		Y = this->GetCommonVariable(Ci, Cj);
		if (Ci->y == Y){
		X = Ci->x;		
		} else {
			X = Ci->y;			
		}
		if (Cj->y == Y){			
			Z = Cj->x;
		} else {
			Z = Cj->y;
		}
		
	} else {
		// Case 2: Ci and Cj are not neighbors
		// Ci-...-Cl-Cj
		vector <clique *> neighbors;
		if (Cj->parent != Cj) {
			neighbors.push_back(Cj->parent);
		}
		for (clique * C: Cj->children) {
			neighbors.push_back(C);
		}
		
		Cl = Ci;
		
		for (clique * C: neighbors) {
			if (C->name < Ci->name) {
				cliquePairToCheck = pair <clique*, clique*>(C,Ci);
			} else {
				cliquePairToCheck = pair <clique*, clique*>(Ci,C);
			}
			if (this->cliquePairToVariablePair.find(cliquePairToCheck) != this->cliquePairToVariablePair.end()) {
				if (Ci == cliquePairToCheck.first) {
					Cl = cliquePairToCheck.second;
				} else {
					Cl = cliquePairToCheck.first;
				}
				break;
			}
		}
				
		assert(Cl != Ci);
		
		// Scope(Ci,Cl) = {X,Y}
		if (Ci->name < Cl->name) {
			tie(X,Y) = this->cliquePairToVariablePair[pair <clique*, clique*>(Ci,Cl)];
		} else {
			tie(Y,X) = this->cliquePairToVariablePair[pair <clique*, clique*>(Cl,Ci)];
		}			
				
		assert(Ci->x == X or Ci->y == X);
		assert(Cl->x == Y or Cl->y == Y);
				
		if (Cj->x == Y or Cj->y == Y){
			// Case 2a
			// Scope(Cj) = {Y,Z}
			if (Cj->x == Y) {
				Z = Cj->y;
			} else {
				Z = Cj->x;
			}
		} else {
			// Case 2b
			// Scope(Cl,Cj) = {Y,Z}
			if (Cl->name < Cj->name) {
				tie(Y_temp,Z) = this->cliquePairToVariablePair[pair <clique*, clique*>(Cl,Cj)];
			} else {
				tie(Z,Y_temp) = this->cliquePairToVariablePair[pair <clique*, clique*>(Cj,Cl)];
			}
			assert (Y_temp == Y);
		}
	}	
	
	return (tuple <SEM_vertex *,SEM_vertex *,SEM_vertex *>(X,Y,Z));
}

Matrix4f cliqueTree::GetP_XZ(SEM_vertex * X, SEM_vertex * Y, SEM_vertex * Z) {
	Matrix4f P_XY; Matrix4f P_YZ;
	Matrix4f P_ZGivenY; Matrix4f P_XZ;
	
    if (X->id < Y->id) {
		P_XY = this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(X,Y)];
	} else {
		P_XY = this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Y,X)].transpose();
	}
	if (Y->id < Z->id) {
		P_YZ = this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Y,Z)];
	} else {
		P_YZ = this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Z,Y)].transpose();
	}
//	cout << "P_XY is " << endl << P_XY << endl;
//	cout << "P_YZ is " << endl << P_YZ << endl;
	P_ZGivenY = ArrayXXf::Zero(4,4);
	float rowSum;
	for (int row = 0; row < 4; row ++) {		
		rowSum = 0;
		for (int col = 0; col < 4; col ++) {
			rowSum += P_YZ(row,col);
		}
		for (int col = 0; col < 4; col ++) {
			if (rowSum != 0){
				P_ZGivenY(row,col) = P_YZ(row,col)/rowSum;
			}			
		}
	}
	
//	cout << "P_ZGivenY is " << endl << P_ZGivenY << endl;
	
	for (int row = 0; row < 4; row ++) {		
		for (int col = 0; col < 4; col ++) {
			P_XZ(row,col) = 0;
		}
	}
	
	for (int dna_y = 0; dna_y < 4; dna_y ++) {		
		for (int dna_x = 0; dna_x < 4; dna_x ++) {
			for (int dna_z = 0; dna_z < 4; dna_z ++) {					
				// Sum over Y
				P_XZ(dna_x,dna_z) += P_XY(dna_x,dna_y) * P_ZGivenY(dna_y,dna_z);
			}
		}
	}
	
	return (P_XZ);
}

SEM_vertex * cliqueTree::GetCommonVariable(clique * Ci, clique * Cj) {
	SEM_vertex * commonVariable;
	if (Ci->x == Cj->x or Ci->x == Cj->y) {
		commonVariable = Ci->x;
	} else {
		commonVariable = Ci->y;
	}
	return (commonVariable);
}


void cliqueTree::ConstructSortedListOfAllCliquePairs() {
	this->cliquePairsSortedWrtLengthOfShortestPath.clear();
	vector < tuple <int, clique*, clique*>> sortedPathLengthAndCliquePair;
	int pathLength;
	for (clique * Ci : this->cliques) {
		for (clique * Cj : this->cliques) {
			if (Ci->name < Cj->name) {
				if (Ci->outDegree > 0 or Cj->outDegree > 0) {
					pathLength = this->GetDistance(Ci, Cj);
					sortedPathLengthAndCliquePair.push_back(make_tuple(pathLength,Ci,Cj));
				}				
			}
		}
	}
	sort(sortedPathLengthAndCliquePair.begin(),sortedPathLengthAndCliquePair.end());
	clique * Ci; clique * Cj;
	for (tuple <int, clique*, clique*> pathLengthCliquePair : sortedPathLengthAndCliquePair) {
		Ci = get<1>(pathLengthCliquePair);
		Cj = get<2>(pathLengthCliquePair);
		this->cliquePairsSortedWrtLengthOfShortestPath.push_back(pair <clique *, clique *> (Ci,Cj));
	}
}

int cliqueTree::GetDistance(clique * C_1, clique * C_2) {
	clique * lca = this->GetLCA(C_1, C_2);
	int d;
	d = this->GetDistanceToAncestor(C_1,lca) + this->GetDistanceToAncestor(C_2,lca);
	return (d);
}

int cliqueTree::GetDistanceToAncestor(clique * C_d, clique* C_a) {
	int d = 0;
	clique * C_p;
	C_p = C_d;
	while (C_p != C_a) {
		C_p = C_p->parent;
		d += 1;
	}
	return (d);
}

clique * cliqueTree::GetLCA(clique * C_1, clique * C_2) {
	vector <clique *> pathToRootForC1;
	vector <clique *> pathToRootForC2;
	clique * C1_p;
	clique * C2_p;
	C1_p = C_1;
	C2_p = C_2;
	
	clique * C_r = this->edgesForPreOrderTreeTraversal[0].first;
	
	while (C1_p->parent != C1_p) {
		pathToRootForC1.push_back(C1_p);
		C1_p = C1_p->parent;
	}
	pathToRootForC1.push_back(C1_p);
	if (C1_p != C_r) {
		cout << "Check get LCA for C1" << endl;
	}
	
	while (C2_p->parent != C2_p) {
		pathToRootForC2.push_back(C2_p);
		C2_p = C2_p->parent;
	}
	pathToRootForC2.push_back(C2_p);
	if (C2_p != C_r) {
		cout << "Check get LCA for C2" << endl;
	}
	
	clique * lca;
	lca = C_1;
	
	for (clique * C : pathToRootForC1) {
		if (find(pathToRootForC2.begin(),pathToRootForC2.end(),C)!=pathToRootForC2.end()) {
			lca = C;
			break;
		}
	}		
	return (lca);	
}

void cliqueTree::ComputeMarginalProbabilitesForEachEdge() {
	this->marginalizedProbabilitiesForVariablePair.clear();
	//	Store P(X,Y) for each clique
	for (clique * C: this->cliques) {
		if (C->x->id < C->y->id) {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Matrix4f>(pair<SEM_vertex *, SEM_vertex *>(C->x,C->y),C->belief));
		} else {
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Matrix4f>(pair<SEM_vertex *, SEM_vertex *>(C->y,C->x),C->belief.transpose()));			
		}
	}
}

void cliqueTree::ComputeMarginalProbabilitesForEachVariablePair() {	
	this->marginalizedProbabilitiesForVariablePair.clear();
	this->cliquePairToVariablePair.clear();
	// For each clique pair store variable pair 
	// Iterate over clique pairs in order of increasing distance in clique tree	
	
	clique * Ci; clique * Cj;
	
	SEM_vertex * X; SEM_vertex * Z;
	SEM_vertex * Y;

	Matrix4f P_XZ;
		
	//	Store P(X,Y) for each clique
	//	cout << "Number of cliques is " << this->cliques.size() << endl;
	for (clique * C: this->cliques) {
		if (C->x->id < C->y->id) {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Matrix4f>(pair<SEM_vertex *, SEM_vertex *>(C->x,C->y),C->belief));
		} else {
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Matrix4f>(pair<SEM_vertex *, SEM_vertex *>(C->y,C->x),C->belief.transpose()));			
		}
//		if (C->y->name == "l_1") {
//			cout << "belief for site " << this->site << " is " << endl;
//			cout << C->belief << endl;
//		}
	}
	
	for (pair <clique *, clique *> cliquePair : this->cliquePairsSortedWrtLengthOfShortestPath) {
		tie (Ci, Cj) = cliquePair;
	//	cout << "Ci is " << Ci->name << "\t" << "Cj is " << Cj->name << endl;
	//	Scope(Ci,Cj) = {X,Z}
		tie (X, Y, Z) = this->GetXYZ(Ci, Cj);		
	//	cout << "X is " << X->id << "\t,";
	//	cout << "Y is " << Y->id << "\t,";
	//	cout << "Z is " << Z->id << endl;
		this->cliquePairToVariablePair.insert(pair <pair <clique *, clique *>,pair <SEM_vertex *, SEM_vertex *>>(pair <clique *, clique *>(Ci,Cj), pair <SEM_vertex *, SEM_vertex *>(X,Z)));
		P_XZ = this->GetP_XZ(X, Y, Z);
	//	cout << "P_XZ is " << endl << P_XZ << endl;
		if (X->id < Z->id) {
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Matrix4f>(pair<SEM_vertex *, SEM_vertex *>(X,Z),P_XZ));			
		} else {
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Matrix4f>(pair<SEM_vertex *, SEM_vertex *>(Z,X),P_XZ.transpose()));			
		}		
	//	cout << "----------------------------------" << endl;
	}
}


void cliqueTree::SetRoot() {
	for (clique * C: this->cliques) {
		if (C->inDegree == 0) {
			this->root = C;
		}		
	}
}

void cliqueTree::InitializePotentialAndBeliefs() {
	for (clique * C: this->cliques) {		
		C->SetInitialPotentialAndBelief(this->site);
	}
}

void cliqueTree::SetSite(int site) {
	this->site = site;
}

void cliqueTree::AddClique(clique * C) {
	this->cliques.push_back(C);
}

void cliqueTree::AddEdge(clique * C_1, clique * C_2) {
	C_1->AddChild(C_2);
	C_2->AddParent(C_1);
}

void cliqueTree::SetEdgesForTreeTraversalOperations() {
	for (clique * C : this->cliques) {
		C->timesVisited = 0;
	}
	this->edgesForPostOrderTreeTraversal.clear();
	this->edgesForPreOrderTreeTraversal.clear();
	vector <clique *> verticesToVisit;
	verticesToVisit = this->leaves;
	clique * C_child; clique * C_parent;
	int numberOfVerticesToVisit = verticesToVisit.size();
	
	while (numberOfVerticesToVisit > 0) {
		C_child = verticesToVisit[numberOfVerticesToVisit - 1];
		verticesToVisit.pop_back();
		numberOfVerticesToVisit -= 1;
		C_parent = C_child->parent;
		if (C_child != C_parent) {
			C_parent->timesVisited += 1;
			this->edgesForPostOrderTreeTraversal.push_back(make_pair(C_parent, C_child));
			if (C_parent->timesVisited == C_parent->outDegree) {				
				verticesToVisit.push_back(C_parent);
				numberOfVerticesToVisit += 1;				
			}
		}
	}
	
	for (int edgeInd = this->edgesForPostOrderTreeTraversal.size() -1; edgeInd > -1; edgeInd --) {
		this->edgesForPreOrderTreeTraversal.push_back(this->edgesForPostOrderTreeTraversal[edgeInd]);
	}
}

void cliqueTree::SetLeaves() {
	this->leaves.clear();
	for (clique * C: this->cliques) {
		if (C->outDegree == 0) {
			this->leaves.push_back(C);
		}		
	}
}


void cliqueTree::SendMessage(clique * C_from, clique * C_to) {		
	float logScalingFactor;
	float largestElement;	
	array <float, 4> messageFromNeighbor;
	array <float, 4> messageToNeighbor;
	bool verbose = 0;
	if (verbose) {
		cout << "Preparing message to send from " << C_from->name << " to " ;
		cout << C_to->name << " is " << endl;
	}
	
	// Perform the three following actions
	
	// A) Compute product: Multiply the initial potential of C_from
	// with messages from all neighbors of C_from except C_to, and
	
	// B) Compute sum: Marginalize over the variable that
	// is in C_from but not in C_to
	
	// C) Transmit: sending the message to C_to
	
	// Select neighbors
	vector <clique *> neighbors;
	if (C_from->parent != C_from and C_from->parent != C_to) {
		neighbors.push_back(C_from->parent);
	}
	
	for (clique * C_child : C_from->children) {
		if (C_child != C_to) {
			neighbors.push_back(C_child);
		}
	}
	
	Matrix4f factor;
	factor = C_from->initialPotential;
	
	logScalingFactor = 0;
	if (verbose) {
		cout << "Initial potential of " << C_from->name << " is" << endl;
		cout << factor << endl;
	}		
	
	//	PRODUCT: Multiply messages from neighbors that are not C_to
	for (clique * C_neighbor : neighbors) {
		messageFromNeighbor = C_from->messagesFromNeighbors[C_neighbor];
		if (C_from->y == C_neighbor->x or C_from->y == C_neighbor->y) {
//		factor_row_i = factor_row_i (dot) message
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
				for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
					factor(dna_x, dna_y) *= messageFromNeighbor[dna_y];					
				}
			}
			if (verbose) {cout << "Performing row-wise multiplication" << endl;}			
		} else if (C_neighbor->x == C_from->x or C_neighbor->y == C_from->x) {
//		factor_col_i = factor_col_i (dot) message
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
					factor(dna_x, dna_y) *= messageFromNeighbor[dna_x];
				}
			}
			if (verbose) {cout << "Performing column-wise multiplication" << endl;}			
		} else {
			cout << "Check product step" << endl;
			exit (-1);
		}
//		cout << factor << endl;
//		Check to see if each entry in the factor is zero
		bool allZero = 1;
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				if (factor(dna_x,dna_y) == 0) {
					allZero = 0;
				}
			}
		}
		if (allZero and verbose) {
			cout << "All entries in the factor are zero" << endl;
			cout << factor << endl;
		}
//		Rescale factor
		largestElement = 0;
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				if (largestElement < factor(dna_x,dna_y)) {
					largestElement = factor(dna_x,dna_y);
				}
			}
		}
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				factor(dna_x, dna_y) /= largestElement;
			}
		}
		logScalingFactor += log(largestElement);
		logScalingFactor += C_from->logScalingFactorForMessages[C_neighbor];
	}
//	SUM
//		Marginalize factor by summing over common variable
	largestElement = 0;
	if (C_from->y == C_to->x or C_from->y == C_to->y) {
		// Sum over C_from->x		
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			messageToNeighbor[dna_y] = 0;
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
				messageToNeighbor[dna_y] += factor(dna_x, dna_y);
			}
		}
		if (verbose) {
			cout << "Performing column-wise summation" << endl;
		}							
	} else if (C_from->x == C_to->x or C_from->x == C_to->y) {
		// Sum over C_from->y		
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			messageToNeighbor[dna_x] = 0;
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				messageToNeighbor[dna_x] += factor(dna_x, dna_y);
			}
		}
		if (verbose) {
			cout << "Performing row-wise summation" << endl;
		}							
	} else {		
		cout << "Check sum step" << endl;
		exit (-1);
	}
	// Rescale message to neighbor
	largestElement = 0;
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		if (largestElement < messageToNeighbor[dna_x]) {
			largestElement = messageToNeighbor[dna_x];
		}
	}
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		messageToNeighbor[dna_x] /= largestElement;
	}
	logScalingFactor += log(largestElement);
	if (verbose) {
		cout << "Sending the following message from " << C_from->name << " to " ;
		cout << C_to->name << " is " << endl;
		for (int dna = 0; dna < 4; dna ++) {
			cout << messageToNeighbor[dna] << "\t";
			assert(!isnan(messageToNeighbor[dna]));
		}
	}	
	// TRANSMIT
	C_to->logScalingFactorForMessages.insert(make_pair(C_from,logScalingFactor));
	C_to->messagesFromNeighbors.insert(make_pair(C_from,messageToNeighbor));
}

void cliqueTree::CalibrateTree() {
	clique * C_p; clique * C_c;
	//	Send messages from leaves to root
	//	cout << "Sending messages from leaves to root" << endl;
	//	cout << "Number of edges for post order tree traversal is" << endl;
	//	cout << this->edgesForPostOrderTreeTraversal.size() << endl;
	for (pair <clique *, clique *> cliquePair : this->edgesForPostOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		this->SendMessage(C_c, C_p);
	}
	//	Send messages from root to leaves
	//	cout << "Sending messages from root to leaves" << endl;
	//	cout << "Number of edges for pre order tree traversal is" << endl;
	//	cout << this->edgesForPreOrderTreeTraversal.size() << endl;
	for (pair <clique *, clique *> cliquePair : this->edgesForPreOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		this->SendMessage(C_p, C_c);
	}
	//	cout << "Computing beliefs" << endl;
	//	Compute beliefs
	for (clique * C: this->cliques) {
		C->ComputeBelief();		
	}
}


class SEM {
	
public:
	int largestIdOfVertexInMST = 1;
	default_random_engine generator;
	bool setParameters;
	bool verbose = 0;
	string modelForRooting;
	map <string,unsigned char> mapDNAtoInteger;
	map <int, SEM_vertex*> * vertexMap;
	vector <int> sitePatternWeights;
	vector <vector <int> > sitePatternRepetitions;
	vector <int> sortedDeltaGCThresholds;
	int numberOfInputSequences;
	int numberOfVerticesInSubtree;
	int numberOfObservedVertices;
	int numberOfExternalVertices = 0;	
	int numberOfSitePatterns;
	double logLikelihoodConvergenceThreshold = 0.1;
	float sumOfExpectedLogLikelihoods = 0;
	float maxSumOfExpectedLogLikelihoods = 0;
	int h_ind = 1;
	chrono::system_clock::time_point t_start_time;
	chrono::system_clock::time_point t_end_time;
	ofstream logFile;
	SEM_vertex * root;
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForPostOrderTreeTraversal;
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForPreOrderTreeTraversal;	
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForChowLiuTree;
	vector < pair <SEM_vertex *, SEM_vertex *>> directedEdgeList;
	map <pair <SEM_vertex *, SEM_vertex *>, float> edgeLengths;
	vector < SEM_vertex *> leaves;
	vector < SEM_vertex *> preOrderVerticesWithoutLeaves;
	map < pair <SEM_vertex * , SEM_vertex *>, Matrix4f > expectedCountsForVertexPair;
	map < pair <SEM_vertex * , SEM_vertex *>, Matrix4f > posteriorProbabilityForVertexPair;
	map < SEM_vertex *, array <float,4>> expectedCountsForVertex; 
	map < SEM_vertex *, array <float,4>> posteriorProbabilityForVertex;
	map <int, Matrix4f> rateMatrixPerRateCategory;
	map <int, Matrix4f> rateMatrixPerRateCategory_stored;
	map <int, float> scalingFactorPerRateCategory;
	map <int, float> scalingFactorPerRateCategory_stored;
	int numberOfRateCategories = 0;
	float maximumLogLikelihood;
	Matrix4f I4by4;	
	cliqueTree * cliqueT;
	bool debug;
	bool finalIterationOfSEM;
	bool flag_logDet = 0;
	bool flag_Hamming = 0;
	bool flag_JC = 0;
	map <string, int> nameToIdMap;
	string sequenceFileName;
	string ancestralSequencesString = "";
	float sequenceLength;
	// Add vertices (and compressed sequence for leaves)
	array <float, 4> rootProbability;
	array <float, 4> rootProbability_stored;
	SEM_vertex * root_stored;
	vector <unsigned char> compressedSequenceToAddToMST;
	string nameOfSequenceToAddToMST;
	float logLikelihood;	
	// Used for updating MST
	vector <int> indsOfVerticesOfInterest;
	vector <int> indsOfVerticesToKeepInMST;
	vector <int> idsOfVerticesOfInterest;
	vector <int> idsOfObservedVertices;	
	vector <int> idsOfVerticesToRemove;
	vector <int> idsOfVerticesToKeepInMST;	
	vector <int> idsOfExternalVertices;	
	vector < tuple <int, string, vector <unsigned char> > > idAndNameAndSeqTuple;

	// Used for updating global phylogenetic tree
	vector < pair <int, int> > edgesOfInterest_ind;	
	vector < pair < vector <unsigned char>, vector <unsigned char> > > edgesOfInterest_seq;
	string weightedEdgeListString;
	map < string, vector <unsigned char>> sequencesToAddToGlobalPhylogeneticTree;
	vector < tuple <string, string, float>> weightedEdgesToAddToGlobalPhylogeneticTree;
	vector < tuple <string, string, float>> edgeLogLikelihoodsToAddToGlobalPhylogeneticTree;		
	map <string, float> vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree;
	map <pair<SEM_vertex *,SEM_vertex *>,float> edgeLogLikelihoodsMap;
	SEM_vertex * externalVertex;
	void AddArc(SEM_vertex * from, SEM_vertex * to);
	void RemoveArc(SEM_vertex * from, SEM_vertex * to);
	void SetStream(ofstream& stream_to_set);
	void ClearDirectedEdges();
	void ClearUndirectedEdges();
	void ClearAllEdges();
	void AddVertex(string name, vector <unsigned char> compressedSequenceToAdd);
	void AddWeightedEdges(vector<tuple<string,string,float>> weightedEdgesToAdd);
	void AddEdgeLogLikelihoods(vector<tuple<string,string,float>> edgeLogLikelihoodsToAdd);
	void AddExpectedCountMatrices(map < pair <SEM_vertex * , SEM_vertex *>, Matrix4f > expectedCountsForVertexPair);
	void AddVertexLogLikelihoods(map <string,float> vertexLogLikelihoodsMapToAdd);
	void SetNumberOfVerticesInSubtree(int numberOfVertices);	
	void AddSitePatternWeights(vector <int> sitePatternWeightsToAdd);
	void AddSitePatternRepeats(vector <vector <int> > sitePatternRepetitionsToAdd);
	void AddSequences(vector <vector <unsigned char>> sequencesToAdd);	
	void OpenAncestralSequencesFile();
	void AddRootVertex();
//	void AddCompressedSequencesAndNames(map<string,vector<unsigned char>> sequencesList, vector <vector <int>> sitePatternRepeats);
	void AddAllSequences(string sequencesFileName);
	void AddNames(vector <string> namesToAdd);
	void AddGlobalIds(vector <int> idsToAdd);
	// void ComputeNJTreeUsingHammingDistances(); 
	void ComputeNJTree(); // Hamming (default) or LogDet distance
	float ComputeDistance(int v_i, int v_j);
	void RootedTreeAlongAnEdgeIncidentToCentralVertex();
	void RootTreeAlongAnEdgePickedAtRandom();
	void RootTreeAtAVertexPickedAtRandom();
	void RootTreeByFittingAGMMViaEM();
	void RootTreeByFittingUNREST();
	void RootTreeUsingSpecifiedModel(string modelForRooting);
	void RootTreeBySumOfExpectedLogLikelihoods();
	void ComputeSumOfExpectedLogLikelihoods();
	void RootTreeAlongEdge(SEM_vertex * u, SEM_vertex * v);
	void SelectEdgeIncidentToVertexViaMLUnderGMModel(SEM_vertex * v);
	void InitializeTransitionMatricesAndRootProbability();
	void ComputeMAPEstimateOfAncestralSequencesUsingHardEM();
	void ComputeMPEstimateOfAncestralSequences();
	void ComputeMAPEstimateOfAncestralSequences();
	void ComputeMAPEstimateOfAncestralSequencesUsingCliques();
	void RootTreeAtAnEdgeIncidentToVertexThatMaximizesLogLikelihood();
	void SetEdgesForPreOrderTraversal();	
	void SetEdgesForPostOrderTraversal();
	void SetEdgesForTreeTraversalOperations();
	void SetLeaves();
	void SetVerticesForPreOrderTraversalWithoutLeaves();
	void SetObservedUnobservedStatus();
	void OptimizeParametersUsingMAPEstimates();
	void ComputeMLEOfRootProbability();
	void ComputeMLEOfTransitionMatrices();	
	void ComputePosteriorProbabilitiesUsingExpectedCounts();
	void ComputePosteriorProbabilitiesUsingMAPEstimates();
	void SetInfoForVerticesToAddToMST();
	void SetIdsOfExternalVertices();
	void ClearAncestralSequences();
	void RemoveEdgeLength(SEM_vertex * u, SEM_vertex * v);
	void AddEdgeLength(SEM_vertex * u, SEM_vertex * v, float t);
	float GetEdgeLength(SEM_vertex * u, SEM_vertex * v);
	float ComputeEdgeLength(SEM_vertex * u, SEM_vertex * v);
	void SetEdgeLength(SEM_vertex * u, SEM_vertex * v, float t);
	void OptimizeTopology();
	void ComputeChowLiuTree();
	void AddSubforestOfInterest(SEM * localPhylogeneticTree);
	void ReadRootedTree(string treeFileName);
	int GetVertexId(string v_name);
	bool ContainsVertex(string v_name);
	int GetEdgeIndex (int vertexIndex1, int vertexIndex2, int numberOfVertices);
//	SEM_vertex * GetCommonVariable(clique * C_u, clique * C_v);
	Matrix4f GetP_yGivenx(Matrix4f P_xy);
	Matrix4f GetTransitionMatrix(SEM_vertex * p, SEM_vertex * c);
	array <float, 4> GetBaseComposition(SEM_vertex * v);
	array <float, 4> GetObservedCountsForVariable(SEM_vertex * v);
	string modelSelectionCriterion;
	string distance_measure_for_NJ = "Hamming";
	void SetModelSelectionCriterion(string modelSelectionCriterionToSet);
	void RootTreeAtVertex(SEM_vertex * r);
	void StoreEdgeListForChowLiuTree();
	void RestoreEdgeListForChowLiuTree();
	void StoreDirectedEdgeList();
	void RestoreDirectedEdgeList();
	void StoreRootAndRootProbability();
	void RestoreRootAndRootProbability();
	void StoreTransitionMatrices();	
	void RestoreTransitionMatrices();
	void StoreRateMatricesAndScalingFactors();
	void RestoreRateMatricesAndScalingFactors();
	void ResetPointerToRoot();
	void ResetTimesVisited();
	void SetIdsForObservedVertices(vector <int> idsOfObservedVerticesToAdd);
	void SetNumberOfInputSequences(int numOfInputSeqsToSet);
	void ComputeMLRootedTreeForFullStructureSearch();
	void SetNeighborsBasedOnParentChildRelationships();
	void ComputeMLRootedTreeForRootSearchUnderMultiRateMarkovModel();
	void ComputeMLRootedTreeForRootSearchUnderGMM();
	void ComputeMLEstimatesOfGMMGivenExpectedDataCompletion();
	void ComputeMLEstimatesOfMultiRateMMGivenExpectedDataCompletion();
	void OptimizeQAndtForRateCategory(int rateCat);
	void OptimizeQForRateCategory(int rateCat);
	void OptimizetForRateCategory(int rateCat);	
	void SetMinLengthOfEdges();
	void SetParametersForRateMatrixForNelderMead(double x[], int rateCat);
	double GetNegExpectedLogLikelihoodForRateCat(double x[], int rateCat);
	void NelderMeadForOptimizingParametersForRateCat(int rateCat, int n, double start[], double xmin[], 
		 double *ynewlo, double reqmin, double step[], int konvge,
		 int kcount, int *icount, int *numres, int *ifault);
	void FitAGMModelViaHardEM();
	void ComputeInitialEstimateOfModelParameters();
	void TransformRootedTreeToBifurcatingTree();
	void SwapRoot();
	void SuppressRoot();
	bool IsTreeInCanonicalForm();
	void ComputeLogLikelihood();
	void ComputeExpectedLogLikelihood();
	pair <bool, SEM_vertex *> CheckAndRetrieveSingletonHiddenVertex();
	pair <bool, SEM_vertex *> CheckAndRetrieveHiddenVertexWithOutDegreeZeroAndInDegreeOne();
	pair <bool, SEM_vertex *> CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeOne();
	pair <bool, SEM_vertex *> CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeZero();
	pair <bool, SEM_vertex *> CheckAndRetrieveHiddenVertexWithOutDegreeGreaterThanTwo();
	pair <bool, SEM_vertex *> CheckAndRetrieveObservedVertexThatIsTheRoot();
	pair <bool, SEM_vertex *> CheckAndRetrieveObservedVertexThatIsNotALeafAndIsNotTheRoot();
	float GetExpectedMutualInformation(SEM_vertex * u, SEM_vertex * v);
	void ResetLogScalingFactors();
	// Mutual information I(X;Y) is computed using 
	// P(X,Y), P(X), and P(Y), which in turn are computed using
	// MAP estimates
	void InitializeExpectedCounts();
	void InitializeExpectedCountsForEachVariable();
	void InitializeExpectedCountsForEachVariablePair();
	void InitializeExpectedCountsForEachEdge();
	void ResetExpectedCounts();
	void ConstructCliqueTree();
//	void ComputeExpectedCounts();
	void ComputeExpectedCountsForFullStructureSearch();
	void ComputeExpectedCountsForRootSearch();
	Matrix4f GetObservedCounts(SEM_vertex * u, SEM_vertex * v);
	void AddToExpectedCounts();
	void AddToExpectedCountsForEachVariable();
	void AddToExpectedCountsForEachVariablePair();
	Matrix4f GetExpectedCountsForVariablePair(SEM_vertex * u, SEM_vertex * v);
	Matrix4f GetPosteriorProbabilityForVariablePair(SEM_vertex * u, SEM_vertex * v);
	void AddToExpectedCountsForEachEdge();
	// Mutual information I(X;Y) is computing using 
	// P(X,Y), P(X), and P(Y), which in turn are computed using
	// A calibrated clique tree
	// using P(X,Y) = Sum_{H\{X,Y}}{P(X,Y|H\{X,Y},O)}
	// where H is the set of hidden variables and O is the set of observed variables	
	void RootTreeUsingEstimatedParametersViaML();
	void SetFlagForFinalIterationOfSEM();
	void OptimizeTopologyAndParametersOfGMM();	
//	void RootNJTreeViaEM();
	void SetRateCategories(float DeltaGCThreshold);
	void ComputeGCCountsForEachVertex();
	void SetSortedListOfDeltaGCThresholds();
	void OptimizeParametersForMultiRateMarkovModel();
	void InitializeParameters();
	void ComputeTransitionMatrices();
	float ComputeScalingFactor(Matrix4f Q);
	MatrixXf ComputeStationaryDistribution(Matrix4f Q);
	void PerformModelSelection();
	float BIC;
	float AIC;
	void ComputeBIC();
	void ComputeAIC();
	void TestSEM();
	void StoreEdgeListAndSeqToAdd();
	void SelectIndsOfVerticesOfInterestAndEdgesOfInterest();
	void RenameHiddenVerticesInEdgesOfInterestAndSetIdsOfVerticesOfInterest();
	void SetAncestralSequencesString();
	void SetWeightedEdgesToAddToGlobalPhylogeneticTree();	
	void ComputeVertexLogLikelihood(SEM_vertex * v);
	void ComputeEdgeLogLikelihood(SEM_vertex * u, SEM_vertex * v);	
	void SetEdgeAndVertexLogLikelihoods();
	bool IsNumberOfNonSingletonComponentsGreaterThanZero();
	void WriteTree();
	void WriteAncestralSequences();
	void WriteRootedTreeInNewickFormat(string newickFileName);
	void WriteUnrootedTreeInNewickFormat(string newickFileName);
	void WriteCliqueTreeToFile(string cliqueTreeFileName);
	void WriteRootedTreeAsEdgeList(string fileName);
	void WriteUnrootedTreeAsEdgeList(string fileName);
	void ResetData();
	void AddDuplicatedSequencesToRootedTree(MST_tree * M);
	void AddDuplicatedSequencesToUnrootedTree(MST_tree * M);
	// Select vertex for rooting Chow-Liu tree and update edges in T
	// Modify T such that T is a bifurcating tree and likelihood of updated
	// tree is equivalent to the likelihood of T
	SEM (int largestIdOfVertexInMST_toSet, string distance_measure_for_NJ_to_set, bool verbose_flag_to_set) {
		this->distance_measure_for_NJ = distance_measure_for_NJ_to_set;
		if (this->distance_measure_for_NJ == "logDet") {
			this->flag_logDet = 1;
		} else if (this->distance_measure_for_NJ == "Hamming") {
			this->flag_Hamming = 1;
		} else if (this->distance_measure_for_NJ == "Jukes-Cantor") {
			this->flag_JC = 1;
		}
		this->verbose = verbose_flag_to_set;
		this->largestIdOfVertexInMST = largestIdOfVertexInMST_toSet;
		this->h_ind = 1;
		this->vertexMap = new map <int, SEM_vertex *> ;
		// this->vertexName2IdMap = new map <string, int> ;
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		this->generator = default_random_engine(seed);
		this->I4by4 = ArrayXXf::Zero(4,4);
		for (int i = 0; i < 4; i++) {
			this->I4by4(i,i) = 1.0;
		}
		this->cliqueT = new cliqueTree;		
		mapDNAtoInteger["A"] = 0;
		mapDNAtoInteger["C"] = 1;
		mapDNAtoInteger["G"] = 2;
		mapDNAtoInteger["T"] = 3;
		this->finalIterationOfSEM = 0;
	}
	
	~SEM () {
		for (pair <int, SEM_vertex * > idPtrPair : * this->vertexMap) {
			delete idPtrPair.second;
		}
		this->vertexMap->clear();
		delete this->vertexMap;
		// delete this->vertexName2IdMap;
		delete this->cliqueT;
	}
};

void SEM::SetStream(ofstream& stream_to_set){
	this->logFile = stream_to_set;
}

void SEM::AddDuplicatedSequencesToRootedTree(MST_tree * M) {
	// Store dupl seq names in uniq seq vertex
	float t;
	string uniq_seq_name;
	vector <string> dupl_seq_name_vec;
	SEM_vertex * u;
	SEM_vertex * p;
	SEM_vertex * d;
	SEM_vertex * h;
	vector <unsigned char> emptySequence;
	// vector <SEM_vertex *> uniq_vertex_ptr_vec;
	int v_id = this->vertexMap->size() - 1;
	for (pair <string, vector <string> > uniq_seq_name_2_dupl_seq_name_vec : M->unique_seq_id_2_dupl_seq_ids) {
		uniq_seq_name = uniq_seq_name_2_dupl_seq_name_vec.first;
		dupl_seq_name_vec = uniq_seq_name_2_dupl_seq_name_vec.second;
		u = (*this->vertexMap)[this->nameToIdMap[uniq_seq_name]];
		p = u->parent;
		t = this->edgeLengths[make_pair(p,u)];
		
		v_id += 1;
		h = new SEM_vertex(v_id,emptySequence);
		this->vertexMap->insert(pair<int,SEM_vertex*>(h->id,h));
		h->name = "h_" + to_string(this->h_ind);
		this->nameToIdMap.insert(make_pair(h->name,h->id));
		this->h_ind += 1;
		
		u->RemoveParent();
		p->RemoveChild(u);
		this->RemoveEdgeLength(p,u);
		
		h->AddParent(p);
		p->AddChild(h);
		this->AddEdgeLength(p,h,t);

		u->AddParent(h);
		h->AddChild(u);
		this->AddEdgeLength(h,u,0.0);

		for (string dupl_seq_name: dupl_seq_name_vec) {
			v_id += 1;
			d = new SEM_vertex(v_id,emptySequence);
			d->name = dupl_seq_name;
			this->vertexMap->insert(pair<int,SEM_vertex*>(d->id,d));
			this->nameToIdMap.insert(make_pair(d->name,d->id));
			// v->dupl_seq_names.push_back(dupl_seq_name);
			d->AddParent(h);	
			h->AddChild(d);
			this->AddEdgeLength(h,d,0.0);
			// this->edgeLengths[make_pair(h,d)] = 0.0;
		}
	}
}

void SEM::AddDuplicatedSequencesToUnrootedTree(MST_tree * M) {
	// Store dupl seq names in uniq seq vertex
	float t;
	string uniq_seq_name;
	vector <string> dupl_seq_name_vec;
	SEM_vertex * l;
	SEM_vertex * n;
	SEM_vertex * d;
	SEM_vertex * h;
	vector <unsigned char> emptySequence;
	// vector <SEM_vertex *> uniq_vertex_ptr_vec;
	int v_id = this->vertexMap->size() - 1;
	for (pair <string, vector <string> > uniq_seq_name_2_dupl_seq_name_vec : M->unique_seq_id_2_dupl_seq_ids) {
		uniq_seq_name = uniq_seq_name_2_dupl_seq_name_vec.first;
		dupl_seq_name_vec = uniq_seq_name_2_dupl_seq_name_vec.second;
		l = (*this->vertexMap)[this->nameToIdMap[uniq_seq_name]];
		n = l->neighbors[0];
		t = this->edgeLengths[make_pair(n,l)];
		
		v_id += 1;
		h = new SEM_vertex(v_id,emptySequence);
		this->vertexMap->insert(pair<int,SEM_vertex*>(h->id,h));
		h->name = "h_" + to_string(this->h_ind);
		this->nameToIdMap.insert(make_pair(h->name,h->id));
		this->h_ind += 1;
		
		l->RemoveNeighbor(n);
		n->RemoveNeighbor(l);
		this->RemoveEdgeLength(n,l);
		
		h->AddNeighbor(n);
		n->AddNeighbor(h);
		this->AddEdgeLength(n,h,t);

		h->AddNeighbor(l);
		l->AddNeighbor(h);
		this->AddEdgeLength(h,l,0.0);

		for (string dupl_seq_name: dupl_seq_name_vec) {
			v_id += 1;
			d = new SEM_vertex(v_id,emptySequence);
			d->name = dupl_seq_name;
			this->vertexMap->insert(pair<int,SEM_vertex*>(d->id,d));
			this->nameToIdMap.insert(make_pair(d->name,d->id));
			// v->dupl_seq_names.push_back(dupl_seq_name);
			d->AddNeighbor(h);	
			h->AddNeighbor(d);
			this->AddEdgeLength(h,d,0.0);
			// this->edgeLengths[make_pair(h,d)] = 0.0;
		}
	}
}


void SEM::SetEdgeAndVertexLogLikelihoods() {
	SEM_vertex * u;	SEM_vertex * v;
	int u_id; int v_id;	
	this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.clear();
	this->edgeLogLikelihoodsToAddToGlobalPhylogeneticTree.clear();
	for (pair <int, int> edge_ind : this->edgesOfInterest_ind) {
		tie (u_id, v_id) = edge_ind;
		u = (*this->vertexMap)[u_id];
		v = (*this->vertexMap)[v_id];
		if (this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.find(u->name) == this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.end()) {
			this->ComputeVertexLogLikelihood(u);
			this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.insert(pair<string,float>(u->name,u->vertexLogLikelihood));
		}	
		if (this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.find(v->name) == this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.end()) {
			this->ComputeVertexLogLikelihood(v);
			this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.insert(pair<string,float>(v->name,v->vertexLogLikelihood));
		}
		this->ComputeEdgeLogLikelihood(u,v);
		this->ComputeEdgeLogLikelihood(v,u);
	}	
}

void SEM::ComputeVertexLogLikelihood(SEM_vertex * v) {
	array <float, 4> prob = this->posteriorProbabilityForVertex[v];
	array <float, 4> Counts = this->expectedCountsForVertex[v];
	v->vertexLogLikelihood = 0;
	for (int i = 0; i < 4; i ++) {
		if (prob[i] > 0) {
			v->vertexLogLikelihood += (log(prob[i]) * Counts[i]);
		}		
	}	
}


void SEM::ComputeEdgeLogLikelihood(SEM_vertex* x, SEM_vertex * y) {	
	Matrix4f P_xy = this->GetPosteriorProbabilityForVariablePair(x,y);
	Matrix4f P_yGivenx = this->GetP_yGivenx(P_xy);
	Matrix4f Counts = this->GetExpectedCountsForVariablePair(x,y);
	float edgeLogLikelihood = 0;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			if (P_yGivenx(dna_x,dna_y) > 0) {
				edgeLogLikelihood += (log(P_yGivenx(dna_x,dna_y)) * Counts(dna_x,dna_y));
			}
		}
	}
	this->edgeLogLikelihoodsToAddToGlobalPhylogeneticTree.push_back(tuple <string, string, float>(x->name,y->name,edgeLogLikelihood));
}

void SEM::SetWeightedEdgesToAddToGlobalPhylogeneticTree() {
	this->weightedEdgesToAddToGlobalPhylogeneticTree.clear();
	int u_id; int v_id;
	SEM_vertex * u; SEM_vertex * v; 
	string u_name; string v_name;
	float t;	
//	cout << "Adding the following edges to the global phylogenetic tree" << endl;
	for (pair <int, int> edge_ind : this->edgesOfInterest_ind) {
		tie (u_id, v_id) = edge_ind;
		u = (*this->vertexMap)[u_id];
		v = (*this->vertexMap)[v_id];
		u_name = u->name;
		v_name = v->name;
		t = this->ComputeEdgeLength(u,v);
//		cout << u_name << "\t" << v_name << "\t" << t << endl;
		this->weightedEdgesToAddToGlobalPhylogeneticTree.push_back(make_tuple(u_name,v_name,t));
	}
}

void SEM::SetAncestralSequencesString() {
	vector <SEM_vertex *> verticesOfInterest;
	int u_id; int v_id;
	vector <unsigned char> fullSeq;
	string DNAString;
	SEM_vertex * u;	SEM_vertex * v;
	this->ancestralSequencesString = "";
	for (pair <int, int> edge_ind : this->edgesOfInterest_ind) {
		tie(u_id, v_id) = edge_ind;		
		u = (*this->vertexMap)[u_id];		
		v = (*this->vertexMap)[v_id];
		if (!u->observed and find(verticesOfInterest.begin(),verticesOfInterest.end(),u)==verticesOfInterest.end()) {
			fullSeq = DecompressSequence(&(u->compressedSequence),&(this->sitePatternRepetitions));
			DNAString = EncodeAsDNA(fullSeq);	
			this->ancestralSequencesString += ">"; 
			this->ancestralSequencesString += u->name + "\n";
			this->ancestralSequencesString += DNAString + "\n";
			verticesOfInterest.push_back(u);
			
		}		
		if (!v->observed and find(verticesOfInterest.begin(),verticesOfInterest.end(),v)==verticesOfInterest.end()) {
			fullSeq = DecompressSequence(&(v->compressedSequence),&(this->sitePatternRepetitions));
			DNAString = EncodeAsDNA(fullSeq);
			this->ancestralSequencesString += ">"; 
			this->ancestralSequencesString += v->name + "\n";
			this->ancestralSequencesString += DNAString + "\n";
			verticesOfInterest.push_back(v);
		}		
	}	
}


void SEM::SetNeighborsBasedOnParentChildRelationships() {
	this->ClearUndirectedEdges();
	SEM_vertex * p; SEM_vertex * c;
	for (pair<SEM_vertex*, SEM_vertex*> edge : this->edgesForPreOrderTreeTraversal) {
		tie (p, c) = edge;
		p->AddNeighbor(c);
		c->AddNeighbor(p);
	}
}

void SEM::SetIdsForObservedVertices(vector <int> idsOfObservedVerticesToAdd) {
	this->idsOfObservedVertices = idsOfObservedVerticesToAdd;
}

void SEM::SetNumberOfInputSequences(int numOfInputSeqsToSet) {
	this->numberOfInputSequences = numOfInputSeqsToSet;	
}
void SEM::ComputeBIC() {
	this->ComputeLogLikelihood();
	this->BIC = -2.0 * this->logLikelihood;
	float n = this->sequenceLength;
	float numberOfFreeParameters = this->edgesForPostOrderTreeTraversal.size();
	numberOfFreeParameters += 11.0 * (this->numberOfRateCategories);
	bool rootHasDistinctRateCat = 1;
	for (SEM_vertex * v : this->root->children) {
		if (this->root->rateCategory == v->rateCategory) {
			rootHasDistinctRateCat = 0;
		}
	}
	if (rootHasDistinctRateCat) {
		numberOfFreeParameters += 3.0;
	}
	this->BIC += log(n) * numberOfFreeParameters;
}

void SEM::SetModelSelectionCriterion(string modelSelectionCriterionToSet) {
	this->modelSelectionCriterion = modelSelectionCriterionToSet;
}

void SEM::SetFlagForFinalIterationOfSEM() {
	this->finalIterationOfSEM = 1;
}


void SEM::ResetData() {
	for (pair<int,SEM_vertex*> idPtrPair : *this->vertexMap) {
		if (idPtrPair.first != -1) {
			delete idPtrPair.second;
		}
	}
	assert(this->vertexMap->size()==1);
	(*this->vertexMap)[-1]->compressedSequence.clear();	
}

int SEM::GetVertexId(string v_name) {
	SEM_vertex * v;
	int idToReturn = -10;
	for (pair<int,SEM_vertex*> idPtrPair : *this->vertexMap) {
		v = idPtrPair.second;
		if (v->name == v_name){
			idToReturn = v->id;						
		}
	}
	if (idToReturn == -10){
		cout << "Unable to find id for:" << v_name << endl;
	}
	return (idToReturn);
}

void SEM::SuppressRoot() {
	SEM_vertex * c_l;
	SEM_vertex * c_r;
	bool proceed = this->root->outDegree == 2;		
	if (proceed) {		
		c_l = this->root->children[0];		
		c_r = this->root->children[1];		
		c_l->AddNeighbor(c_r);
		c_r->AddNeighbor(c_l);
		c_l->RemoveNeighbor(this->root);
		c_r->RemoveNeighbor(this->root);
		this->RemoveArc(this->root,c_l);
		this->RemoveArc(this->root,c_r);
	}
}

void SEM::SwapRoot() {
	SEM_vertex * root_current;
	SEM_vertex * vertexNamedHRoot;
	vector <SEM_vertex *> childrenOfCurrentRoot;
	vector <SEM_vertex *> childrenOfVertexNamedHRoot;
	int n = this->numberOfObservedVertices;	
	if (this->root->name != "h_root") {
//		this->SetEdgesForPostOrderTraversal();		
//		this->ComputeLogLikelihood();		
		root_current = this->root;
		childrenOfCurrentRoot = root_current->children;
		
		vertexNamedHRoot = (*this->vertexMap)[((2*n)-2)];		
		childrenOfVertexNamedHRoot = vertexNamedHRoot->children;
		
		// Swap children of root
		for (SEM_vertex * c: childrenOfVertexNamedHRoot) {
			this->RemoveArc(vertexNamedHRoot,c);
			this->AddArc(root_current,c);
		}
		
		for (SEM_vertex * c: childrenOfCurrentRoot) {			
			this->RemoveArc(root_current,c);
			this->AddArc(vertexNamedHRoot,c);
		}
		
		vertexNamedHRoot->rootProbability = root_current->rootProbability;
		root_current->transitionMatrix = vertexNamedHRoot->transitionMatrix;
		vertexNamedHRoot->transitionMatrix = this->I4by4;
		
		this->AddArc(vertexNamedHRoot->parent,root_current);
		this->RemoveArc(vertexNamedHRoot->parent,vertexNamedHRoot);
		this->root = vertexNamedHRoot;
		this->SetLeaves();	
		this->SetEdgesForPreOrderTraversal();
		this->SetVerticesForPreOrderTraversalWithoutLeaves();
		this->SetEdgesForPostOrderTraversal();
//		this->ComputeLogLikelihood();
	}	
}

int SEM::GetEdgeIndex (int vertexIndex1, int vertexIndex2, int numberOfVertices) {
	int edgeIndex;
	edgeIndex = numberOfVertices*(numberOfVertices-1)/2;
	edgeIndex -= (numberOfVertices-vertexIndex1)*(numberOfVertices-vertexIndex1-1)/2;
	edgeIndex += vertexIndex2 - vertexIndex1 - 1;
	return edgeIndex;
}

void SEM::ReadRootedTree(string treeFileName) {
	string u_name;
	string v_name;
	int u_id;
	int v_id;
	SEM_vertex * u;
	SEM_vertex * v;
	vector <string> splitLine;
	vector <string> leafNames;
	vector <string> ancestorNames;
	vector <string> nonRootVertexNames;	
	string rootName = "";
	vector <unsigned char> emptySequence;
	v_id = 0;
	ifstream edgeListFile(treeFileName.c_str());
	for (string line; getline(edgeListFile, line);) {
		boost::split(splitLine, line, [](char c){return c == '\t';});
		u_name = splitLine[0];		
		v_name = splitLine[1];
		if (find(nonRootVertexNames.begin(),nonRootVertexNames.end(),v_name) == nonRootVertexNames.end()) {
			nonRootVertexNames.push_back(v_name);
		}		
		if (find(ancestorNames.begin(),ancestorNames.end(),u_name)==ancestorNames.end()) {
			ancestorNames.push_back(u_name);
		}
		if (find(leafNames.begin(),leafNames.end(),v_name)==leafNames.end()) {
			if(!boost::starts_with(v_name, "h_")) {
				leafNames.push_back(v_name);
			}
		}
	}
	for (string name: leafNames) {
		SEM_vertex * v = new SEM_vertex(v_id, emptySequence);
		v->name = name;
		v->observed = 1;
//		cout << "v_name is " << name << endl;
		this->vertexMap->insert(pair<int,SEM_vertex*>(v_id,v));
		v_id += 1;
	}
	// Remove root from ancestor names
	for (string name: ancestorNames) {
		if (find(nonRootVertexNames.begin(),nonRootVertexNames.end(),name)==nonRootVertexNames.end()){
			rootName = name;
		}
	}
	this->numberOfObservedVertices = leafNames.size();
	int n = this->numberOfObservedVertices;			
	// cout << "number of observed sequences is " << this->numberOfObservedSequences << endl;
	// Change root name
	
	ancestorNames.erase(remove(ancestorNames.begin(), ancestorNames.end(), rootName), ancestorNames.end());
	for (string name: ancestorNames) {
		SEM_vertex * v = new SEM_vertex(v_id,emptySequence);
		v->name = name;
//		cout << "v_name is " << name << endl;
		this->vertexMap->insert(pair <int,SEM_vertex*> (v_id,v));
		v_id += 1;
	}
	
//	cout << "v id is " << v_id << endl;
//	cout << "v name is " << (*this->vertexMap)[((2 * n) - 2)]->name << endl;
	this->root = new SEM_vertex (((2 * n) - 2), emptySequence);	
	this->root->name = rootName;
	this->vertexMap->insert(pair <int,SEM_vertex*> (((2 * n) - 2), this->root));
//	cout << "Root name is " << rootName << endl;
//	cout << "Root id is " << (2 * n) - 2 << endl;
//	cout << "Root name is " << (*this->vertexMap)[((2 * n) - 2)]->name << endl;
	edgeListFile.clear();
	edgeListFile.seekg(0, ios::beg);
	for (string line; getline(edgeListFile, line);) {
		boost::split(splitLine, line, [](char c){return c == '\t';});
		u_name = splitLine[0];
		v_name = splitLine[1];
		u_id = this->GetVertexId(u_name);
		v_id = this->GetVertexId(v_name);
		u = (*this->vertexMap)[u_id];
		v = (*this->vertexMap)[v_id];
		u->AddChild(v);
		v->AddParent(u);
	}
	edgeListFile.close();	
	this->SetLeaves();
	cout << "Number of leaves is " << this->leaves.size() << endl;
	this->SetEdgesForPreOrderTraversal();
	cout << "Number of edges for pre order traversal is " << this->edgesForPreOrderTreeTraversal.size() << endl;
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
	cout << "Number of vertices for pre order traversal is " << this->preOrderVerticesWithoutLeaves.size() << endl;
	this->SetEdgesForPostOrderTraversal();
	cout << "Number of edges for post order traversal is " << this->edgesForPostOrderTreeTraversal.size() << endl;
}

bool SEM::IsTreeInCanonicalForm() {
	bool valueToReturn = 1;
	SEM_vertex * v;
	for (pair<int,SEM_vertex*> idPtrPair : *this->vertexMap) {
		v = idPtrPair.second;
		if ((!v->observed) and v->outDegree != 2) {
			valueToReturn = 0;
		}
		if (v->observed and v->outDegree != 0) {
			valueToReturn = 0;
		}
	}
	return (valueToReturn);
}

void SEM::AddAllSequences(string fileName) {
	vector <unsigned char> recodedSequence;
	ifstream inputFile(fileName.c_str());
	string v_name;
	string seq = "";	
	int v_id;
	vector <string> vertexNames;	
	vector <vector <unsigned char>> allSequences;	
	for (string line; getline(inputFile, line );) {
		if (line[0]=='>') {
			if (seq != "") {				
				for (char const dna: seq) {
					recodedSequence.push_back(mapDNAtoInteger[string(1,toupper(dna))]);					
				}				
				v_id = this->GetVertexId(v_name);				
				(*this->vertexMap)[v_id]->compressedSequence = recodedSequence;				
				recodedSequence.clear();
			}
			v_name = line.substr(1,line.length());
//			cout << "v_name is " << v_name << endl;
			seq = "";
		} else {
			seq += line ;
		}
	}
	inputFile.close();
	
	for (char const dna: seq) {
		recodedSequence.push_back(mapDNAtoInteger[string(1,toupper(dna))]);
	}
	
	v_id = this->GetVertexId(v_name);
//	cout << "v_name is " << v_name << endl;
	(*this->vertexMap)[v_id]->compressedSequence = recodedSequence;
	
	int numberOfSites = recodedSequence.size();	
	this->numberOfSitePatterns = numberOfSites;
	this->sequenceLength = numberOfSites;
	recodedSequence.clear();
	
	this->sitePatternWeights.clear();
	
	for (int i = 0; i < numberOfSites; i++) {
		this->sitePatternWeights.push_back(1);
	}	
}

void SEM::ClearAncestralSequences() {
	for (pair <int,SEM_vertex*> idPtrPair : *this->vertexMap) {
		if (!idPtrPair.second->observed) {
			idPtrPair.second->compressedSequence.clear();
		}
	}
}

void SEM::RemoveEdgeLength(SEM_vertex * u, SEM_vertex * v) {
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	this->edgeLengths.erase(vertexPair);
}

void SEM::AddEdgeLength(SEM_vertex * u, SEM_vertex * v, float t) {	
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	this->edgeLengths[vertexPair] = t;
}

float SEM::GetEdgeLength(SEM_vertex * u, SEM_vertex * v) {
	float t;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	t = this->edgeLengths[vertexPair];
	return (t);
}

float SEM::ComputeEdgeLength(SEM_vertex * u, SEM_vertex * v) {
	float t = 0;
	int dna_u; int dna_v; 
	for (int site = 0; site < this->numberOfSitePatterns; site++) {
		dna_u = u->compressedSequence[site];
		dna_v = v->compressedSequence[site];
		if (dna_u != dna_v) {
			t += this->sitePatternWeights[site];
		}
	}
	t /= this->sequenceLength;	
	return (t);
}

void SEM::SetEdgeLength(SEM_vertex * u, SEM_vertex * v, float t) {
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	this->edgeLengths[vertexPair] = t;
}

void SEM::StoreEdgeListAndSeqToAdd() {
	this->weightedEdgeListString = "";		
	this->sequencesToAddToGlobalPhylogeneticTree.clear();
	this->weightedEdgesToAddToGlobalPhylogeneticTree.clear();
	SEM_vertex * u; SEM_vertex * v;	
	float t;	
	for (pair <SEM_vertex *, SEM_vertex *> vertexPair : this->edgesForPostOrderTreeTraversal) {
		tie (u, v) = vertexPair;		
		if (u->parent != u) {
			if (v == this->externalVertex and !this->finalIterationOfSEM) {
				this->compressedSequenceToAddToMST = u->compressedSequence;
				this->nameOfSequenceToAddToMST = u->name;				
			} else {
				t = this->ComputeEdgeLength(u,v);			
//				cout << "Adding edge 1 " << u->name << "\t" << v->name << endl;
				this->weightedEdgeListString += u->name + "\t" + v->name + "\t" + to_string(t) + "\n";
				this->weightedEdgesToAddToGlobalPhylogeneticTree.push_back(make_tuple(u->name,v->name,t));
			}
		}
	}	
	u = this->root->children[0];
	v = this->root->children[1];
	if ((v != this->externalVertex and u!= this->externalVertex) or this->finalIterationOfSEM) {
		t = this->ComputeEdgeLength(u,v);
//		cout << "Adding edge 2 " << u->name << "\t" << v->name << endl;
		this->weightedEdgeListString += u->name + "\t" + v->name + "\t" + to_string(t) + "\n";
		this->weightedEdgesToAddToGlobalPhylogeneticTree.push_back(make_tuple(u->name,v->name,t));
	} else if (u == this->externalVertex) {
		this->compressedSequenceToAddToMST = v->compressedSequence;
		this->nameOfSequenceToAddToMST = v->name;
	} else {
		assert (v == this->externalVertex);
		this->compressedSequenceToAddToMST = u->compressedSequence;
		this->nameOfSequenceToAddToMST = u->name;
	}
//	cout << "Name of external sequence is " << this->externalVertex->name << endl;
//	cout << "Name of sequence to add to MST is " << this->nameOfSequenceToAddToMST << endl;
	// Add sequences of all vertices except the following vertices
	// 1) root, 2) external vertex
	for (pair <int, SEM_vertex * > idPtrPair : * this->vertexMap) {
		u = idPtrPair.second;		
		if (u->parent != u){
			if (u != this->externalVertex) {
				this->sequencesToAddToGlobalPhylogeneticTree[u->name] = u->compressedSequence;
			} else if (this->finalIterationOfSEM) {
				this->sequencesToAddToGlobalPhylogeneticTree[u->name] = u->compressedSequence;
			}
		}		
	}	
}

Matrix4f SEM::GetTransitionMatrix(SEM_vertex * p, SEM_vertex * c) {	
	Matrix4f P = ArrayXXf::Zero(4,4);			
	int dna_p; int dna_c;
	for (int site = 0; site < this->numberOfSitePatterns; site ++) {
		if (p->compressedSequence[site] < 4 && c->compressedSequence[site] < 4) { // FIX_AMB
			dna_p = p->compressedSequence[site];
			dna_c = c->compressedSequence[site];		
			P(dna_p,dna_c) += this->sitePatternWeights[site];	
		}		
	}
//	cout << "Sequence of parent: " << EncodeAsDNA(p->compressedSequence) << endl;
//	cout << "Sequence of child: " << EncodeAsDNA(c->compressedSequence) << endl;
//	cout << "Count matrix is " << P << endl;
	float rowSum;
	for (int i = 0; i < 4; i ++) {
		rowSum = 0;
		for (int j = 0; j < 4; j ++) {
			rowSum += P(i,j);
		}
		for (int j = 0; j < 4; j ++) {
			 P(i,j) /= rowSum;
		}
	}
	return P;
}


void SEM::FitAGMModelViaHardEM() {
	this->ClearAncestralSequences();
	this->ComputeMPEstimateOfAncestralSequences();
	// Iterate till convergence of logLikelihood;
	float currentLogLikelihood = this->logLikelihood;
	int numberOfIterations = 0;
	int maxNumberOfIters = 10;
	bool continueEM = 1;
	cout << this->logLikelihood << endl;
//	cout << "Length of compressed root sequence is " << this->root->compressedSequence.size() << endl;
	while (continueEM and numberOfIterations < maxNumberOfIters) {
		numberOfIterations += 1;
		this->ComputeMLEOfRootProbability();
		this->ComputeMLEOfTransitionMatrices();
		this->ClearAncestralSequences();
		this->ComputeMAPEstimateOfAncestralSequences();
//		cout << "Length of compressed root sequence is " << this->root->compressedSequence.size() << endl;
		if (numberOfIterations < 2 or currentLogLikelihood < this->logLikelihood or abs(currentLogLikelihood - this->logLikelihood) > 0.001){
			continueEM = 1;
		} else {
			continueEM = 0;
		}
		currentLogLikelihood = this->logLikelihood;
//		cout << "current logLikelihood is " << currentLogLikelihood << endl;
	}
}

void SEM::WriteTree() {
	this->WriteRootedTreeAsEdgeList(this->sequenceFileName + ".edges");
	this->WriteRootedTreeInNewickFormat(this->sequenceFileName + ".newick");
}

void SEM::OpenAncestralSequencesFile() {
}

void SEM::WriteAncestralSequences() {		
}

void SEM::WriteRootedTreeInNewickFormat(string newickFileName) {
	vector <SEM_vertex *> verticesToVisit;
	SEM_vertex * c;
	SEM_vertex * p;	
	float edgeLength;	
	for (pair <int, SEM_vertex *> idAndVertex : * this->vertexMap) {
		idAndVertex.second->timesVisited = 0;
		if (idAndVertex.second->children.size() == 0) {
			idAndVertex.second->newickLabel = idAndVertex.second->name;
			verticesToVisit.push_back(idAndVertex.second);
		} else {
			idAndVertex.second->newickLabel = "";
		}
	}
	
	pair <SEM_vertex *, SEM_vertex * > vertexPair;
	unsigned int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		c = verticesToVisit[numberOfVerticesToVisit -1];
		verticesToVisit.pop_back();
		numberOfVerticesToVisit -= 1;
		if (c->parent != c) {
			p = c->parent;
			if (p->id < c->id) {
				vertexPair = make_pair(p,c);
			} else {
				vertexPair = make_pair(c,p);
			}
			p->timesVisited += 1;			
			if (this->edgeLengths.find(vertexPair) == this->edgeLengths.end()) {
				edgeLength = 0.1;
			} else {
				edgeLength = this->edgeLengths[vertexPair];
			}
			if (p->timesVisited == int(p->children.size())) {
				p->newickLabel += "," + c->newickLabel + ":" + to_string(edgeLength) + ")";
				verticesToVisit.push_back(p);
				numberOfVerticesToVisit += 1;
			} else if (p->timesVisited == 1) {
				p->newickLabel += "(" + c->newickLabel + ":" + to_string(edgeLength);
			} else {
				p->newickLabel += "," + c->newickLabel + ":" + to_string(edgeLength);
			}			
		}
	}
	ofstream newickFile;
	newickFile.open(newickFileName);
	newickFile << this->root->newickLabel << ";" << endl;
	newickFile.close();
}

void SEM::WriteCliqueTreeToFile(string cliqueTreeFileName) {
	ofstream cliqueTreeFile;
	cliqueTreeFile.open(cliqueTreeFileName);
	clique * parentClique;
	for (clique * childClique : this->cliqueT->cliques) {
		if (childClique->parent != childClique) {
			parentClique = childClique->parent;
			cliqueTreeFile << parentClique->x->name + "_" + parentClique->y->name +"\t";
			cliqueTreeFile << childClique->x->name + "_" + childClique->y->name << "\t";
			cliqueTreeFile << "0.01" << endl;
		}
	}
	cliqueTreeFile.close();
}

void SEM::WriteUnrootedTreeAsEdgeList(string fileName) {
	ofstream treeFile;
	treeFile.open(fileName);
	SEM_vertex * v;
	float t;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		for (SEM_vertex * n : v->neighbors) {
			if (v->id < n->id) {
				t = this->GetEdgeLength(v,n);
				treeFile << v->name << "\t" << n->name << "\t" << t << endl;
			}
		}
	}
	treeFile.close();
}


void SEM::WriteRootedTreeAsEdgeList(string fileName) {
	ofstream treeFile;
	treeFile.open(fileName);
	float t;
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		if (v != v->parent) {
			t = this->GetEdgeLength(v,v->parent);
			treeFile << v->parent->name << "\t" << v->name << "\t" << t << endl;
		}
	}
	treeFile.close();
}

void SEM::RootTreeAtAVertexPickedAtRandom() {
	cout << "num of observed vertices is " << this->numberOfObservedVertices << endl;	
	int n = this->numberOfObservedVertices;
	uniform_int_distribution <int> distribution_v(n,(2*n-3));
	int v_ind = distribution_v(generator);
	cout << "index of vertex selected for rooting is " << v_ind << endl;
	cout << "number of vertices is " << this->vertexMap->size() << endl;
	SEM_vertex * v = (*this->vertexMap)[v_ind];
	cout << "vertex selected for rooting is " << v->id << endl;
	this->RootTreeAtVertex(v);
	
}

void SEM::RootTreeAlongAnEdgePickedAtRandom() {
	int n = this->numberOfObservedVertices;
//	int numOfVertices = this->vertexMap->size();
	uniform_int_distribution <int> distribution_v(0,(2*n-3));
	int v_ind = distribution_v(generator);
	SEM_vertex * v = (*this->vertexMap)[v_ind];	
	int numOfNeighbors = v->neighbors.size();
//	cout << "Number of neighbors of v are " << numOfNeighbors << endl;
	uniform_int_distribution <int> distribution_u(0,numOfNeighbors-1);	
	int u_ind_in_neighborList = distribution_u(generator);
	SEM_vertex * u = v->neighbors[u_ind_in_neighborList];
//	cout << "Rooting tree along edge ";
//	cout << u->name << "\t" << v->name << endl;
	this->RootTreeAlongEdge(u,v);
}

void SEM::SelectEdgeIncidentToVertexViaMLUnderGMModel(SEM_vertex * v_opt) {
	// Add a new vertex 
	cout << "Current log likelihood is " << this->logLikelihood << endl;
	vector <unsigned char> sequence;
	SEM_vertex * r = new SEM_vertex(-1,sequence);
	this->vertexMap->insert(make_pair(-1,r));
	SEM_vertex * v = v_opt;
	this->root = r;
	vector <pair <SEM_vertex *, SEM_vertex *> > edgesForRooting;
	for (SEM_vertex * n : v->neighbors) {
		edgesForRooting.push_back(make_pair(n,v));
	}
	pair <SEM_vertex *, SEM_vertex *> selectedEdge;
	double maxLogLikelihood = this->logLikelihood; 
	int numberOfEdgesTried = 0;
	for (pair <SEM_vertex *, SEM_vertex *> edge : edgesForRooting) {
		this->RootTreeAlongEdge(edge.first, edge.second);
		this->FitAGMModelViaHardEM();
		numberOfEdgesTried += 1;
		if (this->logLikelihood > maxLogLikelihood or numberOfEdgesTried == 1) {
			selectedEdge = edge;
			maxLogLikelihood = this->logLikelihood;
			this->StoreTransitionMatrices();
			this->StoreRootAndRootProbability();
			this->StoreDirectedEdgeList();
		}
	}
	this->RestoreTransitionMatrices();
	this->RestoreRootAndRootProbability();
	this->RestoreDirectedEdgeList();
	this->logLikelihood = maxLogLikelihood;
}

void SEM::RootTreeAlongEdge(SEM_vertex * u, SEM_vertex * v) {
	// Remove lengths of edges incident to root if necessary
	if (this->root->children.size() == 2) {
		SEM_vertex * c_l = this->root->children[0];
		SEM_vertex * c_r = this->root->children[1];
		this->edgeLengths.erase(make_pair(this->root,c_l));
		this->edgeLengths.erase(make_pair(this->root,c_r));
	}
	this->ClearDirectedEdges();
	SEM_vertex * c;
	this->root->AddChild(u);
	this->root->AddChild(v);
	
	SEM_vertex * c_l = this->root->children[0];
	SEM_vertex * c_r = this->root->children[1];
	this->edgeLengths.insert(make_pair(make_pair(this->root,c_l),0.001));
	this->edgeLengths.insert(make_pair(make_pair(this->root,c_r),0.001));	
	u->AddParent(this->root);
	v->AddParent(this->root);
	vector <SEM_vertex *> verticesToVisit;
	vector <SEM_vertex *> verticesVisited;
	verticesToVisit.push_back(u);
	verticesToVisit.push_back(v);
	verticesVisited.push_back(u);
	verticesVisited.push_back(v);
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {		
		c = verticesToVisit[numberOfVerticesToVisit - 1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(c);
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex * n: c->neighbors) {
			if (find(verticesVisited.begin(),verticesVisited.end(),n)==verticesVisited.end()) {
				verticesToVisit.push_back(n);
				numberOfVerticesToVisit += 1;
				c->AddChild(n);
				n->AddParent(c);
			}
		}
	}	
	this->SetLeaves();
//	cout << "Number of leaves is " << this->leaves.size() << endl;
	this->SetEdgesForPreOrderTraversal();
//	cout << "Number of edges for pre order traversal is " << this->edgesForPreOrderTreeTraversal.size() << endl;
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
//	cout << "Number of vertices for pre order traversal is " << this->preOrderVerticesWithoutLeaves.size() << endl;
	this->SetEdgesForPostOrderTraversal();
//	cout << "Number of edges for post order traversal is " << this->edgesForPostOrderTreeTraversal.size() << endl;
}

void SEM::InitializeTransitionMatricesAndRootProbability() {
	// ASR via MP
	// MLE of transition matrices and root probability
		
	vector <SEM_vertex *> verticesToVisit;

	SEM_vertex * p; int numberOfPossibleStates; int pos;
	map <SEM_vertex *, vector <unsigned char>> VU;
	map <SEM_vertex *, unsigned char> V;
	for (int site = 0; site < this->numberOfSitePatterns; site++) {
		VU.clear(); V.clear();
		// Set VU and V for leaves;
		for (SEM_vertex * v : this->leaves) {
			V[v] = v->compressedSequence[site];
			VU[v].push_back(v->compressedSequence[site]);
		}
		// Set VU for ancestors
		for (int v_ind = preOrderVerticesWithoutLeaves.size()-1; v_ind > -1; v_ind--) {
			p = this->preOrderVerticesWithoutLeaves[v_ind];
			map <unsigned char, int> dnaCount;
			for (unsigned char dna = 0; dna < 4; dna++) {
				dnaCount[dna] = 0;
			}
			for (SEM_vertex * c : p->children) {
				for (unsigned char dna: VU[c]) {
					dnaCount[dna] += 1;
				}
			}
			int maxCount = 0;
			for (pair<unsigned char, int> dnaCountPair: dnaCount) {
				if (dnaCountPair.second > maxCount) {
					maxCount = dnaCountPair.second;
				}
			}			
			for (pair<unsigned char, int> dnaCountPair: dnaCount) {
				if (dnaCountPair.second == maxCount) {
					VU[p].push_back(dnaCountPair.first);					
				}
			}			
		}
		// Set V for ancestors
		for (SEM_vertex * v : this->preOrderVerticesWithoutLeaves) {
			if (v->parent == v) {
			// Set V for root
				if (VU[v].size()==1) {
					V[v] = VU[v][0];
				} else {
					numberOfPossibleStates = VU[v].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V[v] = VU[v][pos];
				}				
			} else {
				p = v->parent;
				if (find(VU[v].begin(),VU[v].end(),V[p])==VU[v].end()){
					numberOfPossibleStates = VU[v].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V[v] = VU[v][pos];					
				} else {
					V[v] = V[p];
				}				
			}
			// Push states to compressedSequence
			v->compressedSequence.push_back(V[v]);
		}
	}
}

void SEM::TestSEM() {
	this->debug = 0;	
	cout << "Testing structural EM" << endl;	
	this->OptimizeTopologyAndParametersOfGMM();
}

void SEM::AddToExpectedCountsForEachVariable() {
	SEM_vertex * v;
	float siteWeight = this->sitePatternWeights[this->cliqueT->site];	
	// Add to counts for each unobserved vertex (C->x) where C is a clique
	array <float, 4> marginalizedProbability;
	vector <SEM_vertex *> vertexList;
	for (clique * C: this->cliqueT->cliques) {
		v = C->x;
		assert(!v->observed);
		if (find(vertexList.begin(),vertexList.end(),v) == vertexList.end()) {
			vertexList.push_back(v);
			marginalizedProbability = C->MarginalizeOverVariable(C->y);
			for (int i = 0; i < 4; i++) {
				this->expectedCountsForVertex[v][i] += marginalizedProbability[i] * siteWeight;
			}
		}
	}
	vertexList.clear();
}

void SEM::AddToExpectedCountsForEachVariablePair() {
	SEM_vertex * u; SEM_vertex * v;
	float siteWeight = this->sitePatternWeights[this->cliqueT->site];	
	pair <SEM_vertex *, SEM_vertex*> vertexPair;
	Matrix4f countMatrixPerSite;
	for (pair<int,SEM_vertex*> idPtrPair_1 : *this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex*> idPtrPair_2 : *this->vertexMap) {
			v = idPtrPair_2.second;			
			if (u->id < v->id) {
				if (!u->observed or !v->observed) {
					vertexPair = pair <SEM_vertex *, SEM_vertex *>(u,v);
					countMatrixPerSite = this->cliqueT->marginalizedProbabilitiesForVariablePair[vertexPair];
					for (int dna_u = 0; dna_u < 4; dna_u ++) {
						for (int dna_v = 0; dna_v < 4; dna_v ++) {
							this->expectedCountsForVertexPair[vertexPair](dna_u,dna_v) += countMatrixPerSite(dna_u, dna_v) * siteWeight;
						}
					}
				}
			}
		}
	}
}

void SEM::AddExpectedCountMatrices(map < pair <SEM_vertex * , SEM_vertex *>, Matrix4f > expectedCountsForVertexPairToAdd) {
	string u_name;
	string v_name;
	SEM_vertex * u;
	SEM_vertex * v;
	pair <SEM_vertex *, SEM_vertex *> edge;
	Matrix4f CountMatrix;
	for (pair <pair <SEM_vertex * , SEM_vertex *>, Matrix4f> mapElem: expectedCountsForVertexPairToAdd) {
		u_name = mapElem.first.first->name;
		v_name = mapElem.first.second->name;
		CountMatrix = mapElem.second;
		u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		if (u->id < v->id) {
			this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(u,v)] = CountMatrix;
		} else {
			this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(v,u)] = CountMatrix.transpose();
		}
	}	//this->expectedCountsForVertexPair
}


Matrix4f SEM::GetExpectedCountsForVariablePair(SEM_vertex * u, SEM_vertex * v) {
	Matrix4f C_pc;
	if (u->id < v->id) {
		C_pc = this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(u,v)];	
	} else {
		C_pc = this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(v,u)].transpose();	
	}
	return (C_pc);
}

Matrix4f SEM::GetPosteriorProbabilityForVariablePair(SEM_vertex * u, SEM_vertex * v) {
	Matrix4f P;
	if (u->id < v->id) {
		P = this->posteriorProbabilityForVertexPair[pair<SEM_vertex *, SEM_vertex *>(u,v)];
	} else {
		P = this->posteriorProbabilityForVertexPair[pair<SEM_vertex *, SEM_vertex *>(v,u)].transpose();
	}
	return (P);
}

void SEM::AddToExpectedCountsForEachEdge() {
	float siteWeight = this->sitePatternWeights[this->cliqueT->site];
	Matrix4f countMatrixPerSite;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	SEM_vertex * u; SEM_vertex * v;
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
		tie (u,v) = edge;
		if (u->id < v->id) {
			vertexPair.first = u; vertexPair.second = v;
		} else {
			vertexPair.second = u; vertexPair.first = v;
		}
		countMatrixPerSite = this->cliqueT->marginalizedProbabilitiesForVariablePair[vertexPair];
		for (int dna_u = 0; dna_u < 4; dna_u ++) {
			for (int dna_v = 0; dna_v < 4; dna_v ++) {
				this->expectedCountsForVertexPair[vertexPair](dna_u,dna_v) += countMatrixPerSite(dna_u, dna_v) * siteWeight;
			}
		}
	}
}

void SEM::AddToExpectedCounts() {
	SEM_vertex * u; SEM_vertex * v;
	float siteWeight = this->sitePatternWeights[this->cliqueT->site];	
	// Add to counts for each unobserved vertex (C->x) where C is a clique
	array <float, 4> marginalizedProbability;
	vector <SEM_vertex *> vertexList;
	for (clique * C: this->cliqueT->cliques) {
		v = C->x;
		assert(!v->observed);
		if (find(vertexList.begin(),vertexList.end(),v) == vertexList.end()) {
			vertexList.push_back(v);
			marginalizedProbability = C->MarginalizeOverVariable(C->y);
			for (int i = 0; i < 4; i++) {
				this->expectedCountsForVertex[v][i] += marginalizedProbability[i] * siteWeight;
			}
		}
	}
	vertexList.clear();
	// Add to counts for each vertex pair
	pair <SEM_vertex *, SEM_vertex*> vertexPair;
	Matrix4f countMatrixPerSite;
	for (pair<int,SEM_vertex*> idPtrPair_1 : *this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex*> idPtrPair_2 : *this->vertexMap) {
			v = idPtrPair_2.second;			
			if (u->id < v->id) {
				if (!u->observed or !v->observed) {
					vertexPair = pair <SEM_vertex *, SEM_vertex *>(u,v);
					countMatrixPerSite = this->cliqueT->marginalizedProbabilitiesForVariablePair[vertexPair];					
//					if (u->name == "l_1" or v->name == "l_1") {
//						cout << "Count matrix for " << u->name << ", " << v->name << " for site " << this->cliqueT->site << " is" << endl;
// 						cout << countMatrixPerSite << endl;
//					}
					for (int dna_u = 0; dna_u < 4; dna_u ++) {
						for (int dna_v = 0; dna_v < 4; dna_v ++) {
							this->expectedCountsForVertexPair[vertexPair](dna_u,dna_v) += countMatrixPerSite(dna_u, dna_v) * siteWeight;
						}
					}
				}
			}
		}
	}
}

Matrix4f SEM::GetObservedCounts(SEM_vertex * u, SEM_vertex * v) {	
	Matrix4f countMatrix = ArrayXXf::Zero(4,4);
	int dna_u; int dna_v;
	for (int i = 0; i < this->sequenceLength; i++) {
		dna_u = u->compressedSequence[i];
		dna_v = v->compressedSequence[i];
		countMatrix(dna_u,dna_v) += this->sitePatternWeights[i];
	}
	return (countMatrix);
}

void SEM::ComputeExpectedCountsForRootSearch() {
//	cout << "Initializing expected counts" << endl;
	this->InitializeExpectedCountsForEachVariable();
	this->InitializeExpectedCountsForEachEdge();
//	this->ResetExpectedCounts();
//	SEM_vertex * x; SEM_vertex * y; 
	Matrix4f P_XY;
//	int dna_x; int dna_y;
	bool debug = 0;
	if (debug) {
		cout << "Debug computing expected counts" << endl;
	}
// Iterate over sites
	// parallelize here if needed
	for (int site = 0; site < this->numberOfSitePatterns; site++) {
		this->cliqueT->SetSite(site);		
		this->cliqueT->InitializePotentialAndBeliefs();		
		this->cliqueT->CalibrateTree();
		this->cliqueT->ComputeMarginalProbabilitesForEachEdge();
		this->AddToExpectedCountsForEachVariable();
		this->AddToExpectedCountsForEachEdge();		
	}
}

void SEM::ComputeMAPEstimateOfAncestralSequencesUsingCliques() {
	this->logLikelihood = 0;
	this->ClearAncestralSequences();
	this->ConstructCliqueTree();
	clique * rootClique = this->cliqueT->root;
	SEM_vertex * v;
	map <SEM_vertex *, int> verticesVisitedMap;
	array <float, 4> posteriorProbability;
	int maxProbState;
	float maxProb;
	for (int site = 0; site < this->numberOfSitePatterns; site++) {		
		this->cliqueT->SetSite(site);		
		this->cliqueT->InitializePotentialAndBeliefs();		
		this->cliqueT->CalibrateTree();
		this->logLikelihood += rootClique->logScalingFactorForClique * this->sitePatternWeights[site];
//		logLikelihood_c0 + = C_1->logScalingFactorForClique * this->sitePatternWeights[site];
//		for (int i = 0; i < 4; i ++) {
//			for (int j = 0; j < 4; j ++) {
//				
//			}
//		}
		verticesVisitedMap.clear();
		for (clique * C: this->cliqueT->cliques) {
			v = C->x;
			if (verticesVisitedMap.find(v) == verticesVisitedMap.end()) {
				posteriorProbability = C->MarginalizeOverVariable(C->y);
				maxProb = -1; maxProbState = -1;
				for (int i = 0; i < 4; i ++) {
					if (posteriorProbability[i] > maxProb) {
						maxProb = posteriorProbability[i];
						maxProbState = i;
					}
				}
				assert(maxProbState != -1);
				v->compressedSequence.push_back(maxProbState);
				verticesVisitedMap.insert(make_pair(v,v->id));
			}
		}
	}	
}


void SEM::ComputeExpectedCountsForFullStructureSearch() {
	bool debug = 0;
//void SEM::ComputeExpectedCounts() {
	if (debug) {
		cout << "Constructing sorted list of all clique pairs" << endl;	
	}
	this->cliqueT->ConstructSortedListOfAllCliquePairs();
//	cout << "Initializing expected counts" << endl;
	if (debug) {
		cout << "Initializing expected counts for each variable" << endl;	
	}
	this->InitializeExpectedCountsForEachVariable();
	if (debug) {
		cout << "Initializing expected counts for each variable pair" << endl;	
	}
	this->InitializeExpectedCountsForEachVariablePair();
//	this->ResetExpectedCounts();
	SEM_vertex * x; SEM_vertex * y; 
	Matrix4f P_XY;
	int dna_x; int dna_y;	
	if (debug) {
		cout << "Debug computing expected counts" << endl;
	}
// Iterate over sites
	for (int site = 0; site < this->numberOfSitePatterns; site++) {				
		if (debug) {
			cout << "Setting site" << endl;	
		}
		this->cliqueT->SetSite(site);		
		if (debug) {
			cout << "Initializing potential and beliefs" << endl;	
		}
		this->cliqueT->InitializePotentialAndBeliefs();		
		if (debug) {
			cout << "Calibrating tree" << endl;	
		}
		this->cliqueT->CalibrateTree();		
		if (debug) {
			cout << "Computing marginal probabilities for each variable pair" << endl;	
		}
		this->cliqueT->ComputeMarginalProbabilitesForEachVariablePair();
		if (debug) {
			cout << "Number of post prob is " << this->cliqueT->marginalizedProbabilitiesForVariablePair.size() << endl; 
			for (pair < pair <SEM_vertex *, SEM_vertex *>, Matrix4f> vertexPairToMatrix : this->cliqueT->marginalizedProbabilitiesForVariablePair) {
				tie (x, y) = vertexPairToMatrix.first;
				P_XY = vertexPairToMatrix.second;		
				cout << "P(X,Y) is " << endl;
				cout << P_XY << endl;			
				dna_x = x->compressedSequence[site];
				dna_y = y->compressedSequence[site];
				cout << "dna_x is " << dna_x;
				cout << " and " << "dna_y is " << dna_y << endl;			
				cout << "P(" << x->name << "," << y->name << ") for site " << site;  
				cout << " is " << P_XY(dna_x,dna_y) << endl;
			}			
		}
		this->AddToExpectedCountsForEachVariable();
		this->AddToExpectedCountsForEachVariablePair();
		// if (debug) {
		// 	break;
		// }
	}
	// Compare observed counts with expected counts (done)
	if (debug) {
		Matrix4f observedCounts;
		Matrix4f expectedCounts;
		for (pair <pair<SEM_vertex *, SEM_vertex *>, Matrix4f> vertexPairToCountMatrix : this->expectedCountsForVertexPair) {
			tie (x,y) = vertexPairToCountMatrix.first;
			expectedCounts = vertexPairToCountMatrix.second;
			observedCounts = this->GetObservedCounts(x,y);
			cout << "Observed count matrix is " << endl;
			cout << observedCounts << endl;
			cout << "Expected count matrix is " << endl;
			cout << expectedCounts << endl;
			cout << "======================" << endl;
		}
	}
}

void SEM::ComputePosteriorProbabilitiesUsingExpectedCounts() {	
	SEM_vertex * v;
	float sum;
	// Compute posterior probability for vertex
	this->posteriorProbabilityForVertex.clear();
	array <float, 4> P_X;	
	for (pair <SEM_vertex *, array <float, 4>> vertexAndCountArray: this->expectedCountsForVertex) {
		v = vertexAndCountArray.first;
		P_X = vertexAndCountArray.second;
		sum = 0;
		for (int i = 0; i < 4; i++) {
			sum += P_X[i];
		}
		for (int i = 0; i < 4; i++) {
			P_X[i] /= sum;
		}
		this->posteriorProbabilityForVertex.insert(pair<SEM_vertex * , array <float, 4>>(v,P_X));
	}
	// Compute posterior probability for vertex pair
	this->posteriorProbabilityForVertexPair.clear();
	Matrix4f P_XY;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	for (pair <pair<SEM_vertex *, SEM_vertex *>, Matrix4f> vertexPairAndCountMatrix: this->expectedCountsForVertexPair) {
		vertexPair = vertexPairAndCountMatrix.first;
		P_XY = vertexPairAndCountMatrix.second;
		sum = 0;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				sum += P_XY(i,j);
			}
		}
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				P_XY(i,j) /= sum;
			}
		}
		this->posteriorProbabilityForVertexPair.insert(pair<pair<SEM_vertex *, SEM_vertex *>,Matrix4f>(vertexPair,P_XY));
	}
}

void SEM::ConstructCliqueTree() {
	this->cliqueT->rootSet = 0;
	for (clique * C : this->cliqueT->cliques) {
		delete C;
	}
	this->cliqueT->cliques.clear();
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
//		cout << edge.first->id << "\t" << edge.second->id << endl;
		clique * C = new clique(edge.first, edge.second);		
		this->cliqueT->AddClique(C);
		if (C->x->parent == C->x and !this->cliqueT->rootSet) {
			this->cliqueT->root = C;
			this->cliqueT->rootSet = 1;
		}
	}
	clique * C_i; clique * C_j;
	// Iterate over clique pairs and identify cliques
	// that have one vertex in common
	for (unsigned int i = 0; i < this->cliqueT->cliques.size(); i ++) {
		C_i = this->cliqueT->cliques[i];
		// Set Ci as the root clique if Ci.x is the root vertex
		for (unsigned int j = i+1; j < this->cliqueT->cliques.size(); j ++) {
			C_j = this->cliqueT->cliques[j];
			// Add edge Ci -> Cj if Ci.y = Cj.x;
			if (C_i->y == C_j->x) {
//				cout << "Case 1" << endl;
//				cout << "C_i.x, C_i.y is " << C_i->x->id << ", " << C_i->y->id << endl;
//				cout << "C_j.x, C_j.y is " << C_j->x->id << ", " << C_j->y->id << endl;
				this->cliqueT->AddEdge(C_i, C_j);
				// Add edge Cj -> Ci if Cj.y = Ci.x;
			} else if (C_j->y == C_i->x) {
//				cout << "Case 2" << endl;
//				cout << "C_i.x, C_i.y is " << C_i->x->id << ", " << C_i->y->id << endl;
//				cout << "C_j.x, C_j.y is " << C_j->x->id << ", " << C_j->y->id << endl;
				this->cliqueT->AddEdge(C_j, C_i);
				// If Ci->x = Cj->x 
				// add edge Ci -> Cj
			} else if (C_i->x == C_j->x and C_i->parent == C_i) {
//				cout << "Case 3" << endl;
//				cout << "C_i.x, C_i.y is " << C_i->x->id << ", " << C_i->y->id << endl;
				this->cliqueT->AddEdge(C_i, C_j);
				// Check to see that Ci is the root clique				
				if (this->cliqueT->root != C_i) {
					cout << "Check root of clique tree" << endl;
					exit(-1);
				}
			}
			// Note that Cj can never be the root clique
			// because Ci is visited before Cj
		}
	}	
	this->cliqueT->SetLeaves();
	this->cliqueT->SetEdgesForTreeTraversalOperations();
}

void SEM::ResetExpectedCounts() {
	SEM_vertex* u; SEM_vertex* v; 
	// Reset counts for each unobserved vertex
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		if (!v->observed) {
			for (int i = 0; i < 4; i++) {
				this->expectedCountsForVertex[v][i] = 0;
			}
		}
	}
	// Reset counts for each vertex pair such that at least one vertex is not observed
	for (pair <int, SEM_vertex *> idPtrPair_1 : * this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex *> idPtrPair_2 : * this->vertexMap) {
			v = idPtrPair_2.second;
			if (!u->observed or !v->observed) {
				if (u->id < v->id) {
					this->expectedCountsForVertexPair[pair <SEM_vertex *, SEM_vertex *>(u,v)] = ArrayXXf::Zero(4,4);
				}	
			}			
		}
	}
}

void SEM::InitializeExpectedCountsForEachVariable() {
	SEM_vertex * v;
	// Initialize expected counts for each vertex
	this->expectedCountsForVertex.clear();
	array <float, 4> observedCounts;
	for (pair<int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		for (int i = 0; i < 4; i++) {
			observedCounts[i] = 0;
		}
		if (v->observed) {			
			observedCounts = this->GetObservedCountsForVariable(v);
		}
		this->expectedCountsForVertex.insert(pair<SEM_vertex *, array<float,4>>(v,observedCounts));
	}	
}

void SEM::InitializeExpectedCountsForEachVariablePair() {
	SEM_vertex * u; SEM_vertex * v;
	// Initialize expected counts for each vertex pair
	this->expectedCountsForVertexPair.clear();	
	Matrix4f countMatrix;	
	int dna_u;
	int dna_v;
	for (pair<int,SEM_vertex *> idPtrPair_1 : * this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex *> idPtrPair_2 : * this->vertexMap) {
			v = idPtrPair_2.second;
			if (u->id < v->id) {
				countMatrix = ArrayXXf::Zero(4,4);			
				if (u->observed and v->observed) {
					for (int site = 0; site < this->numberOfSitePatterns; site++) {
						dna_u = u->compressedSequence[site];
						dna_v = v->compressedSequence[site];
						if (dna_u < 4 && dna_v < 4) { // FIX_AMB
							countMatrix(dna_u,dna_v) += this->sitePatternWeights[site];
						}						
					}
				}
				this->expectedCountsForVertexPair.insert(make_pair(pair <SEM_vertex *, SEM_vertex *>(u,v), countMatrix));
			}
		}
	}
}


void SEM::InitializeExpectedCountsForEachEdge() {
	// Initialize expected counts for each vertex pair
	SEM_vertex * u; SEM_vertex * v;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	this->expectedCountsForVertexPair.clear();
	Matrix4f countMatrix;
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
		countMatrix = ArrayXXf::Zero(4,4);
		tie (u,v) = edge;
		if (u->id < v->id) {
			vertexPair.first = u; vertexPair.second = v;
		} else {
			vertexPair.first = v; vertexPair.second = u;
		}
		this->expectedCountsForVertexPair.insert(make_pair(vertexPair, countMatrix));
	}
}

void SEM::InitializeExpectedCounts() {
	SEM_vertex * u; SEM_vertex * v;
	// Initialize expected counts for each vertex
	this->expectedCountsForVertex.clear();
	array <float, 4> observedCounts;
	for (pair<int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		for (int i = 0; i < 4; i++) {
			observedCounts[i] = 0;
		}
		if (v->observed) {			
			observedCounts = this->GetObservedCountsForVariable(v);
		}
		this->expectedCountsForVertex.insert(pair<SEM_vertex *, array<float,4>>(v,observedCounts));
	}
	
	// Initialize expected counts for each vertex pair
	this->expectedCountsForVertexPair.clear();	
	Matrix4f countMatrix;	
	int dna_u;
	int dna_v;
	for (pair<int,SEM_vertex *> idPtrPair_1 : * this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex *> idPtrPair_2 : * this->vertexMap) {
			v = idPtrPair_2.second;
			if (u->id < v->id) {
				countMatrix = ArrayXXf::Zero(4,4);			
				if (u->observed and v->observed) {
					for (int site = 0; site < this->numberOfSitePatterns; site++) {
						dna_u = u->compressedSequence[site];
						dna_v = v->compressedSequence[site];
						countMatrix(dna_u,dna_v) += this->sitePatternWeights[site];
					}
				}
				this->expectedCountsForVertexPair.insert(make_pair(pair <SEM_vertex *, SEM_vertex *>(u,v), countMatrix));
			}
		}
	}
}

void SEM::ResetPointerToRoot() {
	//	Make sure that the pointer this->root stores the location
	//  of the vertex with in degree 0
	for (pair<int,SEM_vertex *> idPtrPair : *this->vertexMap) {
		if (idPtrPair.second->inDegree == 0){
			this->root = idPtrPair.second;
		}
	}
}

void SEM::AddArc(SEM_vertex * from, SEM_vertex * to) {
	to->AddParent(from);
	from->AddChild(to);
}

void SEM::RemoveArc(SEM_vertex * from, SEM_vertex * to) {	
	to->parent = to;
	to->inDegree -= 1;
	from->outDegree -= 1;	
	int ind = find(from->children.begin(),from->children.end(),to) - from->children.begin();
	from->children.erase(from->children.begin()+ind);	
}

pair<bool,SEM_vertex *> SEM::CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeOne() {
	bool containsVertex = 0;
	SEM_vertex* vPtrToReturn = (*this->vertexMap)[this->vertexMap->size()-1];
	for (pair<int, SEM_vertex *> idPtrPair: *this->vertexMap) {
		if (idPtrPair.second->outDegree == 1 and idPtrPair.second->inDegree == 1) {
			if (idPtrPair.second->id > this->numberOfObservedVertices-1) {
				containsVertex = 1;
				vPtrToReturn = idPtrPair.second;		
				break;
			}
		}		
	}
	return (make_pair(containsVertex,vPtrToReturn));
}

pair <bool,SEM_vertex *> SEM::CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeZero() {
	bool containsVertex = 0;
	SEM_vertex* vPtrToReturn = (*this->vertexMap)[this->vertexMap->size()-1];
	for (pair<int, SEM_vertex*> idPtrPair: *this->vertexMap) {
		if (idPtrPair.second->outDegree == 1 and idPtrPair.second->inDegree == 0) {
			if (idPtrPair.second->id > this->numberOfObservedVertices-1) {
				containsVertex = 1;
				vPtrToReturn = idPtrPair.second;		
				break;
			}
		}		
	}
	return (make_pair(containsVertex,vPtrToReturn));
}

pair <bool, SEM_vertex*> SEM::CheckAndRetrieveSingletonHiddenVertex() {
	bool containsVertex = 0;
	SEM_vertex* vPtrToReturn = (*this->vertexMap)[this->vertexMap->size()-1];
	for (pair<int, SEM_vertex*> idPtrPair: *this->vertexMap) {
		if (!idPtrPair.second->observed and idPtrPair.second->outDegree == 0 and idPtrPair.second->inDegree == 0) {			
			containsVertex = 1;
			vPtrToReturn = idPtrPair.second;		
			break;
		}
	}
	return (make_pair(containsVertex,vPtrToReturn));
}


pair <bool,SEM_vertex*> SEM::CheckAndRetrieveHiddenVertexWithOutDegreeZeroAndInDegreeOne() {
	bool containsVertex = 0;
	SEM_vertex* vPtrToReturn = (*this->vertexMap)[this->vertexMap->size()-1];
	for (pair <int, SEM_vertex*> idPtrPair: *this->vertexMap) {
		if (!idPtrPair.second->observed and idPtrPair.second->outDegree == 0 and idPtrPair.second->inDegree == 1) {
			containsVertex = 1;
			vPtrToReturn = idPtrPair.second;
			break;
		}
	}
	return (make_pair(containsVertex,vPtrToReturn));
}

pair <bool, SEM_vertex *> SEM::CheckAndRetrieveHiddenVertexWithOutDegreeGreaterThanTwo() {
	bool containsVertex = 0;
	SEM_vertex* vPtrToReturn = (*this->vertexMap)[this->vertexMap->size()-1];
	for (pair <int, SEM_vertex *> idPtrPair: *this->vertexMap) {
		if (!idPtrPair.second->observed and idPtrPair.second->outDegree > 2) {
			if (idPtrPair.second->id > this->numberOfObservedVertices-1) {
				containsVertex = 1;
				vPtrToReturn = idPtrPair.second;		
				break;
			}
		}		
	}
	return (make_pair(containsVertex,vPtrToReturn));
}

pair <bool,SEM_vertex *> SEM::CheckAndRetrieveObservedVertexThatIsTheRoot() {
	bool containsVertex = 0;
	SEM_vertex * vPtrToReturn = (*this->vertexMap)[this->vertexMap->size()-1];
	for (pair <int, SEM_vertex *> idPtrPair: *this->vertexMap) {
		if (idPtrPair.second->observed and idPtrPair.second->outDegree > 0) {
			containsVertex = 1;
			vPtrToReturn = idPtrPair.second;
			break;
		}
	}
	return (make_pair(containsVertex,vPtrToReturn));
}

pair <bool,SEM_vertex *> SEM::CheckAndRetrieveObservedVertexThatIsNotALeafAndIsNotTheRoot() {
	bool containsVertex = 0;
	SEM_vertex * vPtrToReturn = (*this->vertexMap)[this->vertexMap->size()-1];
	for (pair <int, SEM_vertex *> idPtrPair: *this->vertexMap) {
		if (idPtrPair.second->observed and idPtrPair.second->outDegree > 0) {
			containsVertex = 1;
			vPtrToReturn = idPtrPair.second;
			break;
		}
	}
	return (make_pair(containsVertex,vPtrToReturn));
}

void SEM::StoreRootAndRootProbability() {
	this->root_stored = this->root;
	this->rootProbability_stored = this->rootProbability;
}

void SEM::RestoreRootAndRootProbability() {
	this->rootProbability = this->rootProbability_stored;
	this->root = this->root_stored;
	this->root->rootProbability = this->rootProbability;	
}

void SEM::StoreTransitionMatrices() {
	for (pair <int,SEM_vertex*> idPtrPair : * this->vertexMap) {
		idPtrPair.second->transitionMatrix_stored = idPtrPair.second->transitionMatrix;
	}
}

void SEM::StoreRateMatricesAndScalingFactors() {
	this->rateMatrixPerRateCategory_stored = this->rateMatrixPerRateCategory;
	this->scalingFactorPerRateCategory_stored = this->scalingFactorPerRateCategory;
}

void SEM::RestoreRateMatricesAndScalingFactors() {
	this->rateMatrixPerRateCategory = this->rateMatrixPerRateCategory_stored;
	this->scalingFactorPerRateCategory = this->scalingFactorPerRateCategory_stored;
}

void SEM::RestoreTransitionMatrices() {
	for (pair<int,SEM_vertex*> idPtrPair : * this->vertexMap) {
		idPtrPair.second->transitionMatrix = idPtrPair.second->transitionMatrix_stored;
	}
}


void SEM::ComputeExpectedLogLikelihood() {
	this->logLikelihood = 0;
	array <float, 4> S_r = this->expectedCountsForVertex[this->root];
//	if (this->root->observed) {
//		cout << "Contribution of observed root " << endl;
//	} else {
//		cout << "Contribution of hidden root " << endl;
//	}
//	for (int i = 0; i < 4; i ++) {
//		cout << S_r[i] << "\t";
//	}
//	cout << endl;
	for (int dna_r = 0; dna_r < 4; dna_r ++) {
		if (this->rootProbability[dna_r] > 0) {
			this->logLikelihood += S_r[dna_r] * log(this->rootProbability[dna_r]);
		}	
	}
//	cout << "vertex log likelihood is " << this->logLikelihood << endl;
	// Contribution of edges
	SEM_vertex * p; SEM_vertex * c;
	Matrix4f S_pc;
	Matrix4f P;
	for (pair <int, SEM_vertex *> idPtrPair : *this->vertexMap) {
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			P = c->transitionMatrix;
			if (p->id < c->id) {
				S_pc = this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(p,c)];	
			} else {
				S_pc = this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(c,p)].transpose();	
			}
			for (int dna_p = 0; dna_p < 4; dna_p ++) {
				for (int dna_c = 0; dna_c < 4; dna_c ++) {
					if(S_pc(dna_p,dna_c) < 0) {
						cout << "Expected counts for " << p->name << "\t" << c->name << " is " << endl;
						cout << S_pc << endl;
						assert(S_pc(dna_p,dna_c) >= 0);
					}
					if (P(dna_p,dna_c) > 0) {
						this->logLikelihood += S_pc(dna_p,dna_c) * log(P(dna_p,dna_c));
					}
				}
			}
		}
	}
//	cout << "total sum of log likelihood is " << this->logLikelihood << endl;
}

// Case 1: Observed vertices may have out degree > 0
// Case 2: Root may have out degree = one
// Case 3: Directed tree (rooted) with vertices with outdegree 2 or 0.
void SEM::ComputeLogLikelihood() {
	this->logLikelihood = 0;
	map <SEM_vertex*,array<float,4>> conditionalLikelihoodMap;
	std::array <float,4> conditionalLikelihood;
	float partialLikelihood;
	float siteLikelihood;	
	float largestConditionalLikelihood = 0;
	float currentProb;			
	vector <SEM_vertex *> verticesToVisit;	
	SEM_vertex * p;
	SEM_vertex * c;
	Matrix4f P;
	for (int site = 0; site < this->numberOfSitePatterns; site++){
		conditionalLikelihoodMap.clear();
		this->ResetLogScalingFactors();
		for (pair<SEM_vertex *,SEM_vertex *> edge : this->edgesForPostOrderTreeTraversal){
			tie (p, c) = edge;					
			P = c->transitionMatrix;	
			p->logScalingFactors += c->logScalingFactors;				
			// Initialize conditional likelihood for leaves
			if (c->outDegree==0) {
				for (unsigned char dna_c = 0; dna_c < 4; dna_c ++){
					conditionalLikelihood[dna_c] = 0;
				}
				conditionalLikelihood[c->compressedSequence[site]] = 1;
				conditionalLikelihoodMap.insert(pair <SEM_vertex *,array<float,4>>(c,conditionalLikelihood));
			}
			// Initialize conditional likelihood for ancestors			
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()){
				// Case 1: Ancestor is not an observed vertex
				if (p->id > this->numberOfObservedVertices -1) {
					for (unsigned char dna_c = 0; dna_c < 4; dna_c++){
						conditionalLikelihood[dna_c] = 1;
					}
				} else {
				// Case 2: Ancestor is an observed vertex
					for (unsigned char dna_c = 0; dna_c < 4; dna_c ++){
						conditionalLikelihood[dna_c] = 0;
					}
					conditionalLikelihood[p->compressedSequence[site]] = 1;
				}								
				conditionalLikelihoodMap.insert(pair <SEM_vertex *,array<float,4>>(p,conditionalLikelihood));					
			}			
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()){
				for (unsigned char dna_c = 0; dna_c < 4; dna_c++){
				conditionalLikelihood[dna_c] = 1;
				}				
				conditionalLikelihoodMap.insert(pair <SEM_vertex *,array<float,4>>(p,conditionalLikelihood));					
			}
			largestConditionalLikelihood = 0;		
			for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
				partialLikelihood = 0;
				for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
//					if (P(dna_p,dna_c)*conditionalLikelihoodMap[c][dna_c] == 0 and P(dna_p,dna_c) > 0 and conditionalLikelihoodMap[c][dna_c] > 0) {
//						cout << "Numerical underflow in computing partial likelihood" << endl;
//						cout << "P(y|x) is " << P(dna_p,dna_c) << endl;
//						cout << "L(y) is " << conditionalLikelihoodMap[c][dna_c] << endl;								
//						cout << "2^-256 is " << 1.0/pow(2,256) << endl;
//					}
					partialLikelihood += P(dna_p,dna_c)*conditionalLikelihoodMap[c][dna_c];
				}
				conditionalLikelihoodMap[p][dna_p] *= partialLikelihood;
				if (conditionalLikelihoodMap[p][dna_p] > largestConditionalLikelihood) {
					largestConditionalLikelihood = conditionalLikelihoodMap[p][dna_p];
				}
			}
			if (largestConditionalLikelihood != 0){
				for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
					conditionalLikelihoodMap[p][dna_p] /= largestConditionalLikelihood;
				}
				p->logScalingFactors += log(largestConditionalLikelihood);
			} else {
				cout << "Largest conditional likelihood value is zero" << endl;
				cout << "Transition matrix P is" << endl;
				cout << "Processing edge " << endl;
				cout << p->name << "\t" << c->name << endl;
				cout << "OutDegrees" << endl;
				cout << p->outDegree << "\t" << c->outDegree << endl;
				cout << P << endl;
				exit(-1);						
			}					
		}
		siteLikelihood = 0; 							
		for (int dna = 0; dna < 4; dna++) {
			currentProb = this->rootProbability[dna]*conditionalLikelihoodMap[this->root][dna];
			siteLikelihood += currentProb;
		}
//		if (site == 0) {
//			cout << "Root probability is" << endl;
//			for (int i = 0; i < 4; i++) {
//				cout << this->rootProbability[i] << "\t";
//			}
//			cout << endl;
//		}
//		if (site == 0) {
//			cout << "LogLikelihood for site 0 is " ;
//			cout << this->root->logScalingFactors + log(siteLikelihood) << endl;
//		}
		this->logLikelihood += (this->root->logScalingFactors + log(siteLikelihood)) * this->sitePatternWeights[site];				
	}
}

void SEM::PerformModelSelection() {		
	cout << "Selecting the number of distinct rate matrices based on BIC" << endl;
	this->RootTreeAtAVertexPickedAtRandom();	
	this->SetSortedListOfDeltaGCThresholds();
//	cout << "Setting sorted list of delta GC thresholds" << endl;	
//	float BIC;
	for (float threshold : this->sortedDeltaGCThresholds) {
//		cout << "Setting rate categories" << endl;
		this->SetRateCategories(threshold);		
//		cout << "Number of rate categories is " << this->numberOfRateCategories << endl;		
//		cout << "Optimizing model parameters " << endl;
 		this->OptimizeParametersForMultiRateMarkovModel();
		this->ComputeLogLikelihood();
		break;
	}
}

void SEM::SetRateCategories(float threshold){
	int rateCat = 0;
	this->numberOfRateCategories = 1;
	SEM_vertex * p; SEM_vertex * c;	
	for (pair<SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
		tie (p, c) = edge;
		if (p->parent == p) {
			p->rateCategory = rateCat;
		}
		c->rateCategory = p->rateCategory;
		if (abs(p->GCContent - c->GCContent) > threshold) {
			c->rateCategory += 1;
			this->numberOfRateCategories += 1;
		}		
	}	
}

void SEM::SetSortedListOfDeltaGCThresholds() {
	this->ComputeMPEstimateOfAncestralSequences();
	this->ComputeGCCountsForEachVertex();
	this->sortedDeltaGCThresholds.clear();
	SEM_vertex * p; SEM_vertex * c; 
	int deltaGC;
	for (pair<int,SEM_vertex*> idPtrPair : * this->vertexMap) {
		c = idPtrPair.second;
		p = c->parent;		
		if (p != c) {
			deltaGC	= abs(p->GCContent - c->GCContent);			
			if (find(this->sortedDeltaGCThresholds.begin(),this->sortedDeltaGCThresholds.end(),deltaGC) == this->sortedDeltaGCThresholds.end()) {
				this->sortedDeltaGCThresholds.push_back(deltaGC);
			}
		}
	}
	sort(this->sortedDeltaGCThresholds.begin(),this->sortedDeltaGCThresholds.end(),greater<int>());
}

void SEM::ComputeGCCountsForEachVertex() {
	SEM_vertex * v;
	int dna;
	for (pair<int,SEM_vertex*> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;		
		v->GCContent = 0;
		for (int site = 0; site < numberOfSitePatterns; site ++) {
			dna = v->compressedSequence[site];
			if (dna == mapDNAtoInteger["G"] or dna == mapDNAtoInteger["C"]) {
				v->GCContent += this->sitePatternWeights[site];
			}
		}		
	}
}

float SEM::ComputeScalingFactor(Matrix4f Q){
	MatrixXf Q_aug = ArrayXXf::Zero(4,5);
	for (int row = 0; row < 4; row++){
		for (int col = 0; col < 4; col++){
			Q_aug(row, col) = Q(row, col);
		}
	}
	for (int row = 0; row < 4; row++){
		Q_aug(row, 4) = 1;
	}	
	MatrixXf b = ArrayXXf::Zero(5,1);
	for (int row = 0; row < 4; row++){
		b(row,0) = 0;
	}
	b(4,0) = 1;	
	MatrixXf pi = ArrayXXf::Zero(1,4);
	pi = Q_aug.transpose().colPivHouseholderQr().solve(b).transpose();
	float scalingFactor = 0;
	for (int i = 0; i < 4; i++){
		scalingFactor -= pi(0,i) * Q(i,i);
	}
	return scalingFactor;
}

MatrixXf SEM::ComputeStationaryDistribution(Matrix4f Q){
	MatrixXf Q_aug = ArrayXXf::Zero(4,5);
	for (int row = 0; row < 4; row++){
		for (int col = 0; col < 4; col++){
			Q_aug(row, col) = Q(row, col);
		}
	}
	for (int row = 0; row < 4; row++){
		Q_aug(row, 4) = 1;
	}	
	MatrixXf b = ArrayXXf::Zero(5,1);
	for (int row = 0; row < 4; row++){
		b(row,0) = 0;
	}
	b(4,0) = 1;	
	MatrixXf pi = ArrayXXf::Zero(4,1);
	pi = Q_aug.transpose().colPivHouseholderQr().solve(b);
	return pi;	
}

void SEM::InitializeParameters() {
	bool resetAllElementsOrRateMatrix = 0;
	map <int, Matrix4f> transitionMatrixForEachCategory;
	Matrix4f P; Matrix4f Q; float scalingFactor;
	int rateCat;
	for (int rateCat = 0 ; rateCat < this->numberOfRateCategories; rateCat ++) {		
		P = ArrayXXf::Zero(4,4);
		for (int i = 0; i < 4; i ++) {
			P(i,i) = 1;
		}
		transitionMatrixForEachCategory.insert(make_pair(rateCat,P));
	}
	SEM_vertex * p; SEM_vertex * c; 
	float sum;
	int dna_p; int dna_c;
	for (pair<SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
		tie (p, c) = edge;
		rateCat = c->rateCategory;
		P = ArrayXXf::Zero(4,4);
		for (int site = 0; site < this->numberOfSitePatterns; site ++) {
			dna_p = p->compressedSequence[site];
			dna_c = c->compressedSequence[site];
			P(dna_p,dna_c) += this->sitePatternWeights[site];
		}
		
		for (int i = 0; i < 4; i ++){
			sum = 0;
			for (int j = 0; j < 4; j ++) {
				sum += P(i,j);
			}
			for (int j = 0; j < 4; j ++) {
				P(i,j) /= sum;
			}
		}
		transitionMatrixForEachCategory[rateCat] = transitionMatrixForEachCategory[rateCat] * P;
	}
	// Compute rate matrix as matrix logarithm
	for (int rateCat = 0; rateCat < this->numberOfRateCategories; rateCat ++) {
		P = transitionMatrixForEachCategory[rateCat];
//		cout << "Transition matrix for rate cat " << rateCat << " is " << endl;
//		cout << P << endl;
		Q = P.log();
		for (int row = 0; row < 4; row ++) {
			for (int col = 0; col < 4; col ++) {
				if (row != col){
					if (Q(row,col) < pow(10,-5)) {
						Q(row,col) = pow(10,-5);
					}
				}
			}
		}
		Q /= Q(3,2);
//		cout << Q << endl;
//		cout << "Is element Q(0,1) nan ? " << isnan(Q(0,1));		
		for (int row = 0; row < 4; row ++) {
			for (int col = 0; col < 4; col ++) {
				if (row != col) {
					if (isnan(Q(row,col) or isinf(Q(row,col)))) {
						resetAllElementsOrRateMatrix = 1;
					}
				}
			}
		}
		
		uniform_real_distribution <> distribution(0.0, 1.0);	
		if (resetAllElementsOrRateMatrix) {
			for (int row = 0; row < 4; row ++) {
				for (int col = 0; col < 4; col ++) {
					if (row != col){						
						Q(row,col) = distribution(generator);
					}
				}
			}		
		}		
		Q /= Q(3,2);				
		Q(0,0) = - (Q(0,1) + Q(0,2) + Q(0,3));
		Q(1,1) = - (Q(1,0) + Q(1,2) + Q(1,3));
		Q(2,2) = - (Q(2,0) + Q(2,1) + Q(2,3));
		Q(3,3) = - (Q(3,0) + Q(3,1) + Q(3,2));
		cout << "Initialized rate matrix for rate cat " << rateCat << " is " << endl;
		cout << Q << endl;
		this->rateMatrixPerRateCategory.insert(make_pair(rateCat,Q));
		scalingFactor = this->ComputeScalingFactor(Q);
		this->scalingFactorPerRateCategory.insert(make_pair(rateCat,scalingFactor));
	}
	Q = this->rateMatrixPerRateCategory[0];
	MatrixXf pi_root = this->ComputeStationaryDistribution(Q);
	for (int i = 0; i < 4; i ++) {
		this->rootProbability[i] = pi_root(i);		
	}	
	this->root->rootProbability = this->rootProbability;	 
}

void SEM::ComputeTransitionMatrices() {
	float scalingFactor; Matrix4f Q; Matrix4f P; float t;
	Matrix4f Q_scaled;
	SEM_vertex * c; SEM_vertex * p;
	for (pair<int,SEM_vertex*> idPtrPair : *this->vertexMap) {
		c = idPtrPair.second;
//		cout << c->name << "\t" << c->parent->name << endl;
		if (c->parent != c) {
			Q = this->rateMatrixPerRateCategory[c->rateCategory];
			p = c->parent;
			t = this->GetEdgeLength(p,c);
//			cout << "Length of edge : ";
//			cout << p->name << "\t" << c->name << endl;
//			cout << " is " << t << endl;
			scalingFactor = this->scalingFactorPerRateCategory[c->rateCategory];
			Q_scaled = Q * (t/scalingFactor);
			P = Q_scaled.exp();
			c->transitionMatrix = P;
		}
	}
}
	

void SEM::OptimizeParametersForMultiRateMarkovModel() {
	double logLikelihood_current;
	int iter = 0;
	int maxIter = 100;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	this->InitializeParameters();
	this->ClearAncestralSequences();
	logLikelihood_current = 0;
//		this->ComputeLogLikelihood();
//		cout << "Initial loglikelihood is " << setprecision(8) << this->logLikelihood << endl;		
	while (continueIterations) {
		iter += 1;
//			cout << "Iteration no. " << iter << endl;
//			cliqueTreeFileName = cliqueTreeFileNamePrefix + "_iter_" +to_string(iter);
//			chowLiuTreeFileName = chowLiuTreeFileNamePrefix + "_iter_" +to_string(iter);
//			MLRootedTreeFileName = MLRootedTreeFileNamePrefix + "_iter_" +to_string(iter);						
//		1. Compute transition matrices
		if (verbose) {
			cout << "Compute transition matrices" << endl;
		}		
		this->ComputeTransitionMatrices();
//		1. Construct clique tree		
		if (verbose) {
			cout << "Construct clique tree" << endl;
		}		
		this->ConstructCliqueTree();				
//			this->WriteCliqueTreeToFile(cliqueTreeFileName);
//		2. Compute expected counts
		if (verbose) {
			cout << "Compute expected counts" << endl;
		}	
		this->ComputeExpectedCountsForRootSearch();
//		this->WriteUnrootedTreeAsEdgeList(chowLiuTreeFileName);
		for (pair< pair <SEM_vertex *, SEM_vertex *>, Matrix4f> edgeAndCount : this->expectedCountsForVertexPair) {
			for (int i = 0; i < 4; i ++) {
				for (int j = 0; j < 4; j ++) {
					assert(edgeAndCount.second(i,j) >= 0);
				}
			}			
		}
//		3. Optimize model parameters
		if (verbose) {
			cout << "Optimize model parameters given expected counts" << endl;
		}		
		this->ComputeMLRootedTreeForRootSearchUnderMultiRateMarkovModel();
//		if (0) {
//			this->WriteRootedTreeAsEdgeList(MLRootedTreeFileName);
//		4. Repeat steps 1 through 3 till convergence	
		if (verbose) {
			cout << "Expected loglikelihood for iteration " << iter << " is " << this->logLikelihood << endl;
		}		
		if ((this->logLikelihood > logLikelihood_current and (abs(this->logLikelihood - logLikelihood_current) > this->logLikelihoodConvergenceThreshold)) or (iter < 2 and iter < maxIter)) {
			logLikelihood_current = this->logLikelihood;
		} else {
			continueIterations = 0;
		}
//		}
//	 continueIterations = 0;
	}
}
void SEM::OptimizeTopologyAndParametersOfGMM() {
	double logLikelihood_current;
	int iter = 0;
	int maxIter = 100;	
	bool continueIterations = 1;
	this->debug = 0;
	this->ComputeNJTree();	
	// perform tree search under ME/BME 
	this->RootTreeAlongAnEdgePickedAtRandom();
	// check is number of hidden vertices equals number of observed vertices -1;
	int nL = 0;
	int nH = 0;
	for (pair <int,SEM_vertex *> idPtrPair : *this->vertexMap) {
		if (idPtrPair.second->observed){
			nL += 1;
		} else {
			nH += 1;
		}
	}
	assert (nH == nL-1);
//	cout << "Number of hidden vertices equals " << nH << endl;
//	cout << "Number of observed vertices equals " << nL << endl;
//		cout << "Initial estimate of ancestral sequences via MP" << endl;
	this->ComputeMPEstimateOfAncestralSequences();	
//		cout << "Initial estimate of model parameters for fully labeled tree" << endl;
	this->ComputeInitialEstimateOfModelParameters();
	this->ClearAncestralSequences();
	logLikelihood_current = 0;
//		this->ComputeLogLikelihood();
//		cout << "Initial loglikelihood is " << setprecision(8) << this->logLikelihood << endl;		
	while (continueIterations) {
		iter += 1;
//			cout << "Iteration no. " << iter << endl;
//			cliqueTreeFileName = cliqueTreeFileNamePrefix + "_iter_" +to_string(iter);
//			chowLiuTreeFileName = chowLiuTreeFileNamePrefix + "_iter_" +to_string(iter);
//			MLRootedTreeFileName = MLRootedTreeFileNamePrefix + "_iter_" +to_string(iter);						
		// 1. Construct clique tree
		if (debug) {
			cout << "Construct clique tree" << endl;
		}
		this->ConstructCliqueTree();
//			this->WriteCliqueTreeToFile(cliqueTreeFileName);
		// 2. Compute expected counts
		if (debug) {
			cout << "Compute expected counts" << endl;
		}
		this->ComputeExpectedCountsForFullStructureSearch();
		// Compare expected counts with actual counts
		// 3. Compute posterior probabilities using expected counts
		if (debug) {
			cout << "Compute posterior probabilities" << endl;
		}
		this->ComputePosteriorProbabilitiesUsingExpectedCounts();
		// 4. Compute Chow-Liu tree
		if (debug) {
			cout << "Compute Chow-Liu tree" << endl;
		}
		this->ComputeChowLiuTree();
//			this->WriteUnrootedTreeAsEdgeList(chowLiuTreeFileName);
		// 5. Transform to ML tree s.t. each the out degree of each vertex is either zero or two.
		if (debug) {
			cout << "Transform to ML tree" << endl;
		}
		this->ComputeMLRootedTreeForFullStructureSearch();
//			this->WriteRootedTreeAsEdgeList(MLRootedTreeFileName);
		// 6. Repeat steps 1 through 6 till convergence		
		if (debug) {
			cout << "Loglikelihood for iteration " << iter << " is " << this->logLikelihood << endl;
		}
		if ((this->logLikelihood > logLikelihood_current and (abs(this->logLikelihood - logLikelihood_current) > this->logLikelihoodConvergenceThreshold)) or (iter < 2 and iter < maxIter)) {
			logLikelihood_current = this->logLikelihood;
		} else {
			continueIterations = 0;
		}	
	}
	// Swap root and "h_root" if they are not identical
//	this->ComputeLogLikelihood();
//	cout << "Marginal loglikelihood after SEM iterations is" << this->logLikelihood << endl;
	this->SwapRoot();	
	// Replace following step with clique tree calibration
	// and marginalizing clique belief		
	this->ComputeMAPEstimateOfAncestralSequencesUsingCliques();	
//	cout << "Log likelihood computed using clique tree is " << this->logLikelihood << endl;
	this->ComputeLogLikelihood();
//	cout << "Log likelihood computed using tree pruning algorithm is " << this->logLikelihood << endl;
//	cout << "Finished computing MAP estimates using cliques" << endl;
	this->SetNeighborsBasedOnParentChildRelationships();	
//	cout << "Computed MAP estimate of ancestral sequences" << endl;
	if ((this->numberOfObservedVertices - this->numberOfVerticesInSubtree) == 1) {
		this->StoreEdgeListAndSeqToAdd();
	}	
}

void SEM::RootTreeByFittingAGMMViaEM() {
//	cout << "10a" << endl;
	this->RootTreeAtAVertexPickedAtRandom();	
//	cout << "10b" << endl;
	this->ComputeMPEstimateOfAncestralSequences();	
//	cout << "10c" << endl;
	this->ComputeInitialEstimateOfModelParameters();
//	cout << "10d" << endl;
//	string cliqueTreeFileNamePrefix = "/home/pk/Projects/MSTBasedForests/data/trees/cliqueTree_test_numberOfLeaves_16_replicate_1";
//	string cliqueTreeFileName;
	this->ClearAncestralSequences();
	double logLikelihood_current;
	int iter = 0;
	int maxIter = 100;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;	
	logLikelihood_current = 0;
//		this->ComputeLogLikelihood();
		cout << "Initial loglikelihood is " << setprecision(8) << this->logLikelihood << endl;
		this->logFile << "Initial loglikelihood is " << setprecision(8) << this->logLikelihood << endl;
	while (continueIterations) {
		t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		cout << "Iteration no. " << iter << endl;
		this->logFile << "Iteration no. " << iter << endl;
//			cliqueTreeFileName = cliqueTreeFileNamePrefix + "_iter_" +to_string(iter);
//			chowLiuTreeFileName = chowLiuTreeFileNamePrefix + "_iter_" +to_string(iter);
//			MLRootedTreeFileName = MLRootedTreeFileNamePrefix + "_iter_" +to_string(iter);						
//		1. Construct clique tree		
		if (verbose) {
			cout << "Construct clique tree" << endl;
			this->logFile << "Iteration no. " << iter << endl;
			
		}		
		this->ConstructCliqueTree();				
//		this->WriteCliqueTreeToFile(cliqueTreeFileName);
//		2. Compute expected counts
		if (verbose) {
			cout << "Compute expected counts" << endl;
		}	
		this->ComputeExpectedCountsForRootSearch();
		this->ComputePosteriorProbabilitiesUsingExpectedCounts();
//		this->WriteUnrootedTreeAsEdgeList(chowLiuTreeFileName);
		for (pair< pair <SEM_vertex *, SEM_vertex *>, Matrix4f> edgeAndCount : this->expectedCountsForVertexPair) {
			for (int i = 0; i < 4; i ++) {
				for (int j = 0; j < 4; j ++) {
					assert(edgeAndCount.second(i,j) >= 0);
				}
			}			
		}
//		3. Optimize model parameters
		if (verbose) {
			cout << "Optimize model parameters given expected counts" << endl;
		}		
		this->ComputeMLRootedTreeForRootSearchUnderGMM();
//		4. Repeat steps 1 through 3 till convergence	
		if (verbose) {
			cout << "Expected loglikelihood for iteration " << iter << " is " << this->logLikelihood << endl;
		}		
		if ((this->logLikelihood > logLikelihood_current and (abs(this->logLikelihood - logLikelihood_current) > this->logLikelihoodConvergenceThreshold)) or (iter < 2 and iter < maxIter)) {
			logLikelihood_current = this->logLikelihood;
		} else {
			continueIterations = 0;
		}
//		}
//	 continueIterations = 0;
	}
	this->ComputeLogLikelihood();	
}



void SEM::RootTreeBySumOfExpectedLogLikelihoods() {
	SEM_vertex * v;
	SEM_vertex * vertexForRooting = (*this->vertexMap)[0];
	int verticesVisited = 0;
//	cout << "Number of edge log likelihoods = " << this->edgeLogLikelihoodsMap.size() << endl;
//	cout << "Number of edge lengths = " << this->edgeLengths.size() << endl;
//	cout << "Number of vertices = " << this->vertexMap->size() << endl;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		if (!v->observed) {
			verticesVisited += 1;
//			cout << v->name << endl;
			this->RootTreeAtVertex(v);
			this->ComputeSumOfExpectedLogLikelihoods();			
//			cout << this->sumOfExpectedLogLikelihoods << endl;
			if ((this->maxSumOfExpectedLogLikelihoods < this->sumOfExpectedLogLikelihoods) or (verticesVisited < 2)){
				this->maxSumOfExpectedLogLikelihoods = this->sumOfExpectedLogLikelihoods;
				vertexForRooting = v;
//				cout << "max expected log likelihood is" << endl;
//				cout << this->maxSumOfExpectedLogLikelihoods << endl;
			}
		}
	}	
	this->RootTreeAtVertex(vertexForRooting);	
}

void SEM::RootTreeUsingSpecifiedModel(string modelForRooting) {
	this->modelForRooting = modelForRooting;
	if (modelForRooting == "GMM") {
		this->RootTreeByFittingAGMMViaEM();
	} else if (modelForRooting == "UNREST") {
		this->RootTreeByFittingUNREST();
	}
}

void SEM::RootTreeByFittingUNREST() {
	// Fit using expected counts
	SEM_vertex * v;
	// Fit for leaf-labeled tree
	for (pair <int, SEM_vertex * > vertElem : *this->vertexMap) {
		v = vertElem.second;
		this->RootTreeAtVertex(v);		
	}		
} 

void SEM::ComputeSumOfExpectedLogLikelihoods() {
	this->sumOfExpectedLogLikelihoods = 0;
	this->sumOfExpectedLogLikelihoods += this->root->vertexLogLikelihood;
	float edgeLogLikelihood;	
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPostOrderTreeTraversal) {
		if (this->edgeLogLikelihoodsMap.find(edge) == this->edgeLogLikelihoodsMap.end()) {
//			cout << edge.first->name << "\t" << edge.second->name << endl;
		} else {
//			cout << edge.first->name << "\t" << edge.second->name << endl;
			edgeLogLikelihood = this->edgeLogLikelihoodsMap[edge];
			this->sumOfExpectedLogLikelihoods += edgeLogLikelihood;
		}				
	}
}

void SEM::ComputeMLRootedTreeForRootSearchUnderGMM() {
	vector < SEM_vertex *> verticesToVisit = this->preOrderVerticesWithoutLeaves;	
	float logLikelihood_max = 0;
	int numberOfVerticesVisited = 0;	
	for (SEM_vertex * v : verticesToVisit) {
		numberOfVerticesVisited += 1;
		this->RootTreeAtVertex(v);		
		this->ComputeMLEstimatesOfGMMGivenExpectedDataCompletion();
		this->ComputeExpectedLogLikelihood();
		if ((numberOfVerticesVisited < 2) or (logLikelihood_max < this->logLikelihood)) {
			logLikelihood_max = this->logLikelihood;
			this->StoreRootAndRootProbability();
			this->StoreTransitionMatrices();			
			this->StoreDirectedEdgeList();
		}
	}
	this->RestoreRootAndRootProbability();
	this->RestoreTransitionMatrices();
	this->RestoreDirectedEdgeList();
	this->SetEdgesForTreeTraversalOperations();
	this->logLikelihood = logLikelihood_max;
}

void SEM::ComputeMLRootedTreeForRootSearchUnderMultiRateMarkovModel() {
	vector < SEM_vertex *> verticesToVisit = this->preOrderVerticesWithoutLeaves;	
	float logLikelihood_max = 0;
	int numberOfVerticesVisited = 0;	
	for (SEM_vertex * v : verticesToVisit) {
		numberOfVerticesVisited += 1;
		this->RootTreeAtVertex(v);
		// Optimize Q and t
		this->ComputeMLEstimatesOfMultiRateMMGivenExpectedDataCompletion();
		this->ComputeExpectedLogLikelihood();
		if ((numberOfVerticesVisited < 2) or (logLikelihood_max < this->logLikelihood)) {
			logLikelihood_max = this->logLikelihood;
			this->StoreRootAndRootProbability();
			this->StoreRateMatricesAndScalingFactors();
			this->StoreDirectedEdgeList();
		}
	}
	this->RestoreRootAndRootProbability();
	this->RestoreRateMatricesAndScalingFactors();
	this->RestoreDirectedEdgeList();
	this->SetEdgesForTreeTraversalOperations();
	this->logLikelihood = logLikelihood_max;
}

void SEM::ComputeMLRootedTreeForFullStructureSearch() {
	this->StoreEdgeListForChowLiuTree();
	// For each vertex v of the Chow-Liu tree
	SEM_vertex * v;
	float logLikelihood_max = 0;
	int verticesTried = 0;
	bool debug = 0;
	bool useExpectedLogLikForSelectingRoot = 1;
	string nonCanonicalRootedTreeFileName = "/home/pk/Projects/MSTBasedForests/data/trees/nonCanonicalRootedTree_test_numberOfLeaves_16_replicate_1";
	string canonicalRootedTreeFileName = "/home/pk/Projects/MSTBasedForests/data/trees/canonicalRootedTree_test_numberOfLeaves_16_replicate_1";
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		verticesTried += 1;
		// 	Root tree at v
		v = idPtrPair.second;
		if (debug) {
			cout << "Rooting tree at vertex " << v->name << endl;
		}
		this->RootTreeAtVertex(v);		
		if (debug) {
			this->WriteRootedTreeAsEdgeList(nonCanonicalRootedTreeFileName);
			cout << "Is v an observed variable?" << endl;
			if (v->observed) {
				cout << "Yes" << endl;
			} else {
				cout << "No" << endl;
			}
			cout << "Root name is " << this->root->name << endl;
		}
		// Compute MLE of model parameters
		// assuming that posterior probabilities
		// P(V) (for each vertex) and 
		// P(V1, V2) (for each vertex pair) are available
//		cout << "Computing MLE of model parameters" << endl;
		this->ComputeMLEstimatesOfGMMGivenExpectedDataCompletion();
		// Transform to bifurcating rooted tree
//		cout << "Transforming to bifurcating leaf-labeled tree" << endl;		
		if (useExpectedLogLikForSelectingRoot) {
			this->ComputeExpectedLogLikelihood();
		} else {
			this->TransformRootedTreeToBifurcatingTree();
			if (debug) {
				this->WriteRootedTreeAsEdgeList(canonicalRootedTreeFileName);
			}	
			// Compute loglikelihood
			this->SetLeaves();
			this->SetEdgesForPostOrderTraversal();
			this->ComputeLogLikelihood();
		}
		if (logLikelihood_max < this->logLikelihood or verticesTried < 2) {
			logLikelihood_max = this->logLikelihood;
//			cout << "Current max loglikelihood is " << logLikelihood_max << endl;
			// Store root probability that maximizes loglikelihood
			this->StoreRootAndRootProbability();
			// Store transition matrices that maximize loglikelihood
			this->StoreTransitionMatrices();
			// Store directed edge list rooted tree which maximizes loglikelihood
			this->StoreDirectedEdgeList();
		}
//		this->RestoreEdgeListForChowLiuTree();
	}
	// Select bifurcating rooted tree and parameters that maximize loglikelihood	
	this->RestoreRootAndRootProbability();
	this->RestoreTransitionMatrices();
	this->RestoreDirectedEdgeList();
	if (useExpectedLogLikForSelectingRoot) {
		this->TransformRootedTreeToBifurcatingTree();
	}
	this->SetLeaves();
	this->SetEdgesForPostOrderTraversal();
	this->SetEdgesForPreOrderTraversal();
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
//	this->ComputeLogLikelihood();
//	Following step computes MLE of parameters of general Markov model	
//	this->logLikelihood = logLikelihood_max;
//	cout << "Current max expected logLikelihood is " << this->logLikelihood << endl;
//	this->ComputeMLEstimatesViaHardEM();
//	cout << "Current logLikelihood is " << this->logLikelihood << endl;
}

Matrix4f SEM::GetP_yGivenx(Matrix4f P_xy) {
	Matrix4f P_yGivenx = ArrayXXf::Zero(4,4);
	array <float, 4> P_x;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		P_x[dna_x] = 0;
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			P_x[dna_x] += P_xy(dna_x, dna_y);
		}
	}
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			P_yGivenx(dna_x, dna_y) = P_xy(dna_x, dna_y) / P_x[dna_x];
		}
	}
	return (P_yGivenx);
}

void SEM::OptimizeQForRateCategory(int rateCat) {
	int i;
	int icount;
	int ifault;
	int kcount;
	int konvge;
	int n;
	int numres;
	double reqmin;
	double *start;
	double *step;
	double *xmin;
	double ynewlo;
	
	n = 11;

	start = new double[n];
	step = new double[n];
	xmin = new double[n];
	
	Matrix4f Q = this->rateMatrixPerRateCategory[rateCat];
	// Initial estimate of parameters
	for (i = 0; i < n; i ++){
		// a
		start[0] = Q(0,1);
		// b
		start[1] = Q(0,2);
		// c
		start[2] = Q(0,3);
		// d
		start[3] = Q(1,0);
		// e
		start[4] = Q(1,2);
		// f
		start[5] = Q(1,3);
		// g
		start[6] = Q(2,0);
		// h
		start[7] = Q(2,1);
		// i
		start[8] = Q(2,3);
		// j
		start[9] = Q(3,0);
		// k
		start[10] = Q(3,1);
	}
		
	reqmin = 1.0E-08;
	double stepSize = pow(10,-2);
    for (i = 0; i < n; i++){
		step[i] = stepSize;
	}
	
	konvge = 10;
	kcount = 500;	
	ynewlo = this->GetNegExpectedLogLikelihoodForRateCat(start, rateCat);	
	this->NelderMeadForOptimizingParametersForRateCat(rateCat, n, start, xmin, &ynewlo, reqmin, step,
	konvge, kcount, &icount, &numres, &ifault);
}

void SEM::SetParametersForRateMatrixForNelderMead(double x[], int rateCat) {	
	Matrix4f Q;
	// a
	Q(0,1) = x[0];
	// b
	Q(0,2) = x[1];
	// c
	Q(0,3) = x[2];
	// d
	Q(1,0) = x[3];
	// e
	Q(1,2) = x[4];
	// f
	Q(1,3) = x[5];
	// g
	Q(2,0) = x[6];
	// h
	Q(2,1) = x[7];
	// i
	Q(2,3) = x[8];
	// j
	Q(3,0) = x[9];
	// k
	Q(3,1) = x[10];
	// set l to 1
	Q(3,2) = 1.0;
	// D1 
	Q(0,0) = - (Q(0,1) + Q(0,2) + Q(0,3));
	// D2 
	Q(1,1) = - (Q(1,0) + Q(1,2) + Q(1,3));
	// D3 
	Q(2,2) = - (Q(2,0) + Q(2,1) + Q(2,3));
	// D4 
	Q(3,3) = - (Q(3,0) + Q(3,1) + Q(3,2));
	
	this->rateMatrixPerRateCategory[rateCat] = Q;
	float scalingFactor = this->ComputeScalingFactor(Q);
	this->scalingFactorPerRateCategory[rateCat] = scalingFactor;
	if (this->root->rateCategory == rateCat) {
		MatrixXf stationaryDistribution = this->ComputeStationaryDistribution(Q);
		for (int i = 0; i < 4; i++) {
			if (stationaryDistribution(i,0) < 0) {
				cout << "Stationary distribution has negative entry" << endl;
				cout << "Rate matrix is " << endl << Q << endl;
			}
			this->rootProbability[i] = stationaryDistribution(i,0);			
		}
		this->root->rootProbability = this->rootProbability;
	}
	return;
}

double SEM::GetNegExpectedLogLikelihoodForRateCat(double x[], int rateCat) {
	double valueToReturn;
	bool setParameters = 1;
	for (int par = 0; par < 11; par++) {
		if (x[par] < pow(10,-4) or x[par] > pow(10,4)) {
			setParameters = 0;
			valueToReturn = pow(10,20);
		}
	}
	if (setParameters) {
		this->SetParametersForRateMatrixForNelderMead(x, rateCat);
		this->ComputeTransitionMatrices();
		this->ComputeExpectedLogLikelihood();
		valueToReturn = -1 * this->logLikelihood;
	}
	return (valueToReturn);
}


void SEM::NelderMeadForOptimizingParametersForRateCat( int rateCat, int n, double start[], double xmin[], 
  double *ynewlo, double reqmin, double step[], int konvge, int kcount, 
  int *icount, int *numres, int *ifault )

//****************************************************************************80
//
//  Purpose:
//
//    NELMIN minimizes a function using the Nelder-Mead algorithm.
//
//  Discussion:
//
//    This routine seeks the minimum value of a user-specified function.
//
//    Simplex function minimisation procedure due to Nelder+Mead(1965),
//    as implemented by O'Neill(1971, Appl.Statist. 20, 338-45), with
//    subsequent comments by Chambers+Ertel(1974, 23, 250-1), Benyon(1976,
//    25, 97) and Hill(1978, 27, 380-2)
//
//    The function to be minimized must be defined by a function of
//    the form
//
//      function fn ( x, f )
//      double fn
//      double x(*)
//
//    and the name of this subroutine must be declared EXTERNAL in the
//    calling routine and passed as the argument FN.
//
//    This routine does not include a termination test using the
//    fitting of a quadratic surface.
//
//  Licensing:
//
//    This code is distributed under the GNU LGPL license. 
//
//  Modified:
//
//    27 February 2008
//
//  Author:
//
//    Original FORTRAN77 version by R ONeill.
//    C++ version by John Burkardt.
//
//  Reference:
//
//    John Nelder, Roger Mead,
//    A simplex method for function minimization,
//    Computer Journal,
//    Volume 7, 1965, pages 308-313.
//
//    R ONeill,
//    Algorithm AS 47:
//    Function Minimization Using a Simplex Procedure,
//    Applied Statistics,
//    Volume 20, Number 3, 1971, pages 338-345.
//
//  Parameters:
//
//    Input, double FN ( double x[] ), the name of the routine which evaluates
//    the function to be minimized.
//
//    Input, int N, the number of variables.
//
//    Input/output, double START[N].  On input, a starting point
//    for the iteration.  On output, this data may have been overwritten.
//
//    Output, double XMIN[N], the coordinates of the point which
//    is estimated to minimize the function.
//
//    Output, double YNEWLO, the minimum value of the function.
//
//    Input, double REQMIN, the terminating limit for the variance
//    of function values.
//
//    Input, double STEP[N], determines the size and shape of the
//    initial simplex.  The relative magnitudes of its elements should reflect
//    the units of the variables.
//
//    Input, int KONVGE, the convergence check is carried out 
//    every KONVGE iterations.
//
//    Input, int KCOUNT, the maximum number of function 
//    evaluations.
//
//    Output, int *ICOUNT, the number of function evaluations 
//    used.
//
//    Output, int *NUMRES, the number of restarts.
//
//    Output, int *IFAULT, error indicator.
//    0, no errors detected.
//    1, REQMIN, N, or KONVGE has an illegal value.
//    2, iteration terminated because KCOUNT was exceeded without convergence.
//
{
  double ccoeff = 0.5;
  double del;
  double dn;
  double dnn;
  double ecoeff = 2.0;
  double eps = 0.001;
  int i;
  int ihi;
  int ilo;
  int j;
  int jcount;
  int l;
  int nn;
  double *p;
  double *p2star;
  double *pbar;
  double *pstar;
  double rcoeff = 1.0;
  double rq;
  double x;
  double *y;
  double y2star;
  double ylo;
  double ystar;
  double z;
//
//  Check the input parameters.
//
  if ( reqmin <= 0.0 )
  {
    *ifault = 1;
    return;
  }

  if ( n < 1 )
  {
    *ifault = 1;
    return;
  }

  if ( konvge < 1 )
  {
    *ifault = 1;
    return;
  }

  p = new double[n*(n+1)];
  pstar = new double[n];
  p2star = new double[n];
  pbar = new double[n];
  y = new double[n+1];

  *icount = 0;
  *numres = 0;

  jcount = konvge; 
  dn = ( double ) ( n );
  nn = n + 1;
  dnn = ( double ) ( nn );
  del = 1.0;
  rq = reqmin * dn;
//
//  Initial or restarted loop.
//
  for ( ; ; )
  {
    for ( i = 0; i < n; i++ )
    { 
      p[i+n*n] = start[i];
    }
    y[n] = this->GetNegExpectedLogLikelihoodForRateCat( start , rateCat);
    *icount = *icount + 1;

    for ( j = 0; j < n; j++ )
    {
      x = start[j];
      start[j] = start[j] + step[j] * del;
      for ( i = 0; i < n; i++ )
      {
        p[i+j*n] = start[i];
      }
      y[j] = this->GetNegExpectedLogLikelihoodForRateCat( start , rateCat);
      *icount = *icount + 1;
      start[j] = x;
    }
//                    
//  The simplex construction is complete.
//                    
//  Find highest and lowest Y values.  YNEWLO = Y(IHI) indicates
//  the vertex of the simplex to be replaced.
//                
    ylo = y[0];
    ilo = 0;

    for ( i = 1; i < nn; i++ )
    {
      if ( y[i] < ylo )
      {
        ylo = y[i];
        ilo = i;
      }
    }
//
//  Inner loop.
//
    for ( ; ; )
    {
      if ( kcount <= *icount )
      {
        break;
      }
      *ynewlo = y[0];
      ihi = 0;

      for ( i = 1; i < nn; i++ )
      {
        if ( *ynewlo < y[i] )
        {
          *ynewlo = y[i];
          ihi = i;
        }
      }
//
//  Calculate PBAR, the centroid of the simplex vertices
//  excepting the vertex with Y value YNEWLO.
//
      for ( i = 0; i < n; i++ )
      {
        z = 0.0;
        for ( j = 0; j < nn; j++ )
        { 
          z = z + p[i+j*n];
        }
        z = z - p[i+ihi*n];  
        pbar[i] = z / dn;
      }
//
//  Reflection through the centroid.
//
      for ( i = 0; i < n; i++ )
      {
        pstar[i] = pbar[i] + rcoeff * ( pbar[i] - p[i+ihi*n] );
      }
      ystar = this->GetNegExpectedLogLikelihoodForRateCat( pstar , rateCat);;
      *icount = *icount + 1;
//
//  Successful reflection, so extension.
//
      if ( ystar < ylo )
      {
        for ( i = 0; i < n; i++ )
        {
          p2star[i] = pbar[i] + ecoeff * ( pstar[i] - pbar[i] );
        }
		y2star = this->GetNegExpectedLogLikelihoodForRateCat( p2star , rateCat);
//        y2star = this->powell ( p2star );
        *icount = *icount + 1;
//
//  Check extension.
//
        if ( ystar < y2star )
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = pstar[i];
          }
          y[ihi] = ystar;
        }
//
//  Retain extension or contraction.
//
        else
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = p2star[i];
          }
          y[ihi] = y2star;
        }
      }
//
//  No extension.
//
      else
      {
        l = 0;
        for ( i = 0; i < nn; i++ )
        {
          if ( ystar < y[i] )
          {
            l = l + 1;
          }
        }

        if ( 1 < l )
        {
          for ( i = 0; i < n; i++ )
          {
            p[i+ihi*n] = pstar[i];
          }
          y[ihi] = ystar;
        }
//
//  Contraction on the Y(IHI) side of the centroid.
//
        else if ( l == 0 )
        {
          for ( i = 0; i < n; i++ )
          {
            p2star[i] = pbar[i] + ccoeff * ( p[i+ihi*n] - pbar[i] );
          }
		  y2star = this->GetNegExpectedLogLikelihoodForRateCat( p2star , rateCat);
//          y2star = this->powell ( p2star );
          *icount = *icount + 1;
//
//  Contract the whole simplex.
//
          if ( y[ihi] < y2star )
          {
            for ( j = 0; j < nn; j++ )
            {
              for ( i = 0; i < n; i++ )
              {
                p[i+j*n] = ( p[i+j*n] + p[i+ilo*n] ) * 0.5;
                xmin[i] = p[i+j*n];
              }
			  y[j] = this->GetNegExpectedLogLikelihoodForRateCat( xmin , rateCat);
//              y[j] = this->powell ( xmin );
              *icount = *icount + 1;
            }
            ylo = y[0];
            ilo = 0;

            for ( i = 1; i < nn; i++ )
            {
              if ( y[i] < ylo )
              {
                ylo = y[i];
                ilo = i;
              }
            }
            continue;
          }
//
//  Retain contraction.
//
          else
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
        }
//
//  Contraction on the reflection side of the centroid.
//
        else if ( l == 1 )
        {
          for ( i = 0; i < n; i++ )
          {
            p2star[i] = pbar[i] + ccoeff * ( pstar[i] - pbar[i] );
          }
		  y2star = this->GetNegExpectedLogLikelihoodForRateCat( p2star , rateCat);
//          y2star = this->powell ( p2star );
          *icount = *icount + 1;
//
//  Retain reflection?
//
          if ( y2star <= ystar )
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = p2star[i];
            }
            y[ihi] = y2star;
          }
          else
          {
            for ( i = 0; i < n; i++ )
            {
              p[i+ihi*n] = pstar[i];
            }
            y[ihi] = ystar;
          }
        }
      }
//
//  Check if YLO improved.
//
      if ( y[ihi] < ylo )
      {
        ylo = y[ihi];
        ilo = ihi;
      }
      jcount = jcount - 1;

      if ( 0 < jcount )
      {
        continue;
      }
//
//  Check to see if minimum reached.
//
      if ( *icount <= kcount )
      {
        jcount = konvge;

        z = 0.0;
        for ( i = 0; i < nn; i++ )
        {
          z = z + y[i];
        }
        x = z / dnn;

        z = 0.0;
        for ( i = 0; i < nn; i++ )
        {
          z = z + pow ( y[i] - x, 2 );
        }

        if ( z <= rq )
        {
          break;
        }
      }
    }
//
//  Factorial tests to check that YNEWLO is a local minimum.
//
    for ( i = 0; i < n; i++ )
    {
      xmin[i] = p[i+ilo*n];
    }
    *ynewlo = y[ilo];

    if ( kcount < *icount )
    {
      *ifault = 2;
      break;
    }

    *ifault = 0;

    for ( i = 0; i < n; i++ )
    {
      del = step[i] * eps;
      xmin[i] = xmin[i] + del;
	  z = this->GetNegExpectedLogLikelihoodForRateCat( xmin , rateCat);
//      z = this->powell ( xmin );
      *icount = *icount + 1;
      if ( z < *ynewlo )
      {
        *ifault = 2;
        break;
      }
      xmin[i] = xmin[i] - del - del;
	  z = this->GetNegExpectedLogLikelihoodForRateCat( xmin , rateCat);
//      z = this->powell ( xmin );
      *icount = *icount + 1;
      if ( z < *ynewlo )
      {
        *ifault = 2;
        break;
      }
      xmin[i] = xmin[i] + del;
    }

    if ( *ifault == 0 )
    {
      break;
    }
//
//  Restart the procedure.
//
    for ( i = 0; i < n; i++ )
    {
      start[i] = xmin[i];
    }
    del = eps;
    *numres = *numres + 1;
  }
  delete [] p;
  delete [] pstar;
  delete [] p2star;
  delete [] pbar;
  delete [] y;

  return;
}

void SEM::SetMinLengthOfEdges() {	
	for (pair<pair<SEM_vertex * , SEM_vertex * >,float> edgeAndLengthsPair: this->edgeLengths){		
		if (edgeAndLengthsPair.second < pow(10,-7)){
			this->edgeLengths[edgeAndLengthsPair.first]  = pow(10,-7);
		}
	}
}


void SEM::OptimizetForRateCategory(int rateCat) {
	SEM_vertex * p;	
	SEM_vertex * c;	
	// Iterate over edges for rate cat
	Matrix4f Q; Matrix4f Q_norm; Matrix4f Q_scaled; Matrix4f P;	Matrix4f C_pc;
	Matrix4f P_first_der; Matrix4f P_second_der;
	double firstDerivative;
	double secondDerivative;
	double firstDerivative_perSite;
	double secondDerivative_perSite;
	float scalingFactor;
	unsigned char dna_p; unsigned char dna_c;
	float t;
	bool convergenceNotReached;
	float stepSize;
	float proposedChange_float;
	for (pair <int, SEM_vertex*> idPtrPair : * this->vertexMap) {
		c = idPtrPair.second;
		if (c->parent != c){
			if (c->rateCategory == rateCat) {				
				p = c->parent;								
				Q = this->rateMatrixPerRateCategory[c->rateCategory];
				scalingFactor = this->scalingFactorPerRateCategory[c->rateCategory];
				Q_norm = Q/scalingFactor;
				C_pc = this->GetExpectedCountsForVariablePair(p,c);
				for (int i = 0; i < 4; i ++) {
					for (int j = 0; j < 4; j ++) {
						if (C_pc(i,j) < 0) {
							cout << "Expected counts for " << p->name << "\t" << c->name << " is " << endl;
							cout << C_pc << endl;
							assert(C_pc(i,j) >= 0);
						}						
					}
				}		
//				cout << "Expected counts for " << p->name << "\t" << c->name << " is " << endl;
//				cout << C_pc << endl;
				// Iterate till convergence
				convergenceNotReached = 1;
				t = this->GetEdgeLength(p,c);
//				cout << "Edge length before updating is " << t << endl;
				while (convergenceNotReached) {					
					Q_scaled = Q_norm * t;				
					P = Q_scaled.exp();
					P_first_der = Q_norm * P;
					P_second_der = Q_norm * P_first_der;
					// Compute first and second derivatives
					firstDerivative = 0;
					secondDerivative = 0;					
					for (dna_p = 0; dna_p < 4; dna_p ++) {
						for (dna_c = 0; dna_c < 4; dna_c ++) {
							firstDerivative_perSite = P_first_der(dna_p,dna_c)/P(dna_p,dna_c);
							firstDerivative += firstDerivative_perSite * C_pc(dna_p,dna_c);
							secondDerivative_perSite = P_second_der(dna_p,dna_c)/P(dna_p,dna_c) - pow(firstDerivative_perSite,2);
							secondDerivative += secondDerivative_perSite * C_pc(dna_p,dna_c);
						}
					}
					stepSize = 1.0;
					proposedChange_float = firstDerivative/secondDerivative;					
					if (abs(proposedChange_float) < pow(10,-4) or t < pow(10,-5)) {
						convergenceNotReached = 0;		
					} else {
						if (t - proposedChange_float < 0.0) {
//							cout << "Case 1: proposed edge length is negative" << endl;
							while (t - stepSize * (proposedChange_float) < 0.0) {
								stepSize /= 2.0;
							}					
							t -= stepSize * (proposedChange_float);
						} else if (t - proposedChange_float > 1.0) {
							t = pow(10,-5);							
						} else {							
							t -= proposedChange_float;
						}						
					}
				}
				this->SetEdgeLength(p,c,t);
//				cout << "Edge length after updating is " << t << "\n";				
			}
		}
	}	
}

void SEM::OptimizeQAndtForRateCategory(int rateCat) {
	bool convergenceNotReached = 1;
	float logLikelihood_current = 0;
	int maxIter = 100;
	int iter = 0;
	this->ComputeExpectedLogLikelihood();
//	cout << "Initial log likelihood is ";
//	cout << this->logLikelihood << endl;
	while (convergenceNotReached) {
		iter += 1;
		this->OptimizeQForRateCategory(rateCat);
		this->OptimizetForRateCategory(rateCat);
		if ((this->logLikelihood > logLikelihood_current and (abs(this->logLikelihood - logLikelihood_current) > this->logLikelihoodConvergenceThreshold)) or (iter < 2 and iter < maxIter)) {
			logLikelihood_current = this->logLikelihood;
		} else {
			convergenceNotReached = 0;
		}
	}
//	cout << "Final log likelihood is ";
//	cout << this->logLikelihood << endl;
}

void SEM::ComputeMLEstimatesOfMultiRateMMGivenExpectedDataCompletion() {
	for (int rateCat = 0; rateCat < this->numberOfRateCategories; rateCat++) {
		this->OptimizeQAndtForRateCategory(rateCat);
	}
}
 
void SEM::ComputeMLEstimatesOfGMMGivenExpectedDataCompletion() {
	SEM_vertex * x; SEM_vertex * y;
	Matrix4f P_xy;
	for (pair <int, SEM_vertex*> idPtrPair : *this->vertexMap) {
		y = idPtrPair.second;
		x = y->parent;
		if (x != y) {
			if (x->id < y->id) {
				P_xy = this->posteriorProbabilityForVertexPair[make_pair(x,y)];
			} else {
				// Check following step
				P_xy = this->posteriorProbabilityForVertexPair[make_pair(y,x)].transpose();
			}
			// MLE of transition matrices
			y->transitionMatrix = this->GetP_yGivenx(P_xy);
		} else {
			// MLE of root probability
			this->rootProbability = this->posteriorProbabilityForVertex[y];
			y->rootProbability = this->rootProbability;
			y->transitionMatrix = ArrayXXf::Zero(4,4);
			for (int i = 0; i < 4; i ++) {
				y->transitionMatrix(i,i) = 1.0;
			}
		}
	}
}

void SEM::StoreEdgeListForChowLiuTree() {
	this->edgesForChowLiuTree.clear();	
	SEM_vertex * v;
	for(pair <int, SEM_vertex*> idPtrPair : *this->vertexMap) {
		v = idPtrPair.second;
		for (SEM_vertex * n : v->neighbors) {
			if (v->id < n->id) {
				this->edgesForChowLiuTree.push_back(make_pair(v,n));
			} else {
				this->edgesForChowLiuTree.push_back(make_pair(n,v));
			}
		}
	}
}

void SEM::RestoreEdgeListForChowLiuTree() {
	this->ClearDirectedEdges();
	SEM_vertex *u; SEM_vertex *v; 
	for(pair<SEM_vertex *,SEM_vertex*> edge : this->edgesForChowLiuTree){
		tie (u, v) = edge;
		u->AddNeighbor(v);
		v->AddNeighbor(u);
	}
}

void SEM::StoreDirectedEdgeList() {
	SEM_vertex * p;
	SEM_vertex * c;
	this->directedEdgeList.clear();
	for(pair <int, SEM_vertex*> idPtrPair : *this->vertexMap){
		c = idPtrPair.second;
		if (c->parent != c) {			
			p = c->parent;
			this->directedEdgeList.push_back(make_pair(p,c));
		}
	}
}

void SEM::RestoreDirectedEdgeList() {
	this->ClearDirectedEdges();
	SEM_vertex * p;
	SEM_vertex * c;
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->directedEdgeList) {
		tie (p, c) = edge;
		c->AddParent(p);
		p->AddChild(c);		
	}
}

void SEM::RootTreeAtVertex(SEM_vertex* r) {	
	this->ClearDirectedEdges();
	vector <SEM_vertex*> verticesToVisit;
	vector <SEM_vertex*> verticesVisited;
	SEM_vertex * p;	
	verticesToVisit.push_back(r);	
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		p = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(p);
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex* c : p->neighbors) {
			if (find(verticesVisited.begin(),verticesVisited.end(),c)==verticesVisited.end()) {
				p->AddChild(c);
				c->AddParent(p);				
				verticesToVisit.push_back(c);
				numberOfVerticesToVisit += 1;
			}
		}
	}
	this->root = r;
	this->SetEdgesForTreeTraversalOperations();
}

void SEM::TransformRootedTreeToBifurcatingTree() {
	bool containsMatchingVertex;
	bool containsSingletonHiddenVertex;
	bool debug = 0;
	SEM_vertex * matchingVertex;
	SEM_vertex * p; SEM_vertex * c; SEM_vertex * h;
	SEM_vertex * o; SEM_vertex * h_s;
	SEM_vertex * c_1; SEM_vertex * c_2; 
	Matrix4f P;
	array <float, 4> rootProb_orig;
	array <float, 4> rootProb_new;
	vector <SEM_vertex *> childrenToSwap;
	bool checkForCanonicalForm;
	checkForCanonicalForm = IsTreeInCanonicalForm();
	string nonCanonicalRootedTreeFileName = "/home/pk/Projects/MSTBasedForests/data/trees/debugNonCanonicalRootedTree_before";
	this->WriteRootedTreeAsEdgeList(nonCanonicalRootedTreeFileName);
	int numberOfTransformations = 0;
	while (!checkForCanonicalForm) {
		numberOfTransformations += 1;
		if (numberOfTransformations > 10000) {
			cout << "Check Transformation of rooted tree to canonical form" << endl;
			exit(-1);
		}
		// Case 1. x->h
		// Check for hidden vertices that are leaves
		// Remove the arc x->h. This creates a singleton vertex h.
		tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeZeroAndInDegreeOne();		
		while (containsMatchingVertex) {
			if (debug and containsMatchingVertex) {
				cout << "Case 1. There is a hidden vertex that is a leaf" << endl;
			}
			this->RemoveArc(h->parent,h);
			tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeZeroAndInDegreeOne();
		}
		// Case 2. p->h->c
		// Check for hidden vertices h with out degree 1 and in degree 1 
		// Remove p->h and h->c and add p->c
		tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeOne();
		while (containsMatchingVertex) {
			if (debug and containsMatchingVertex) {
				cout << "Case 2. There is a non-root hidden vertex that has out degree 1" << endl;
			}			
			p = h->parent;
			assert (h->children.size() == 1);
			c = h->children[0];
//			cout << "h is " << h->name << "\t";
//			cout << "p is " << p->name << "\t";
//			cout << "c is " << c->name << endl;
			// Add edge (p,c)						
//			cout << "Current transition matrices are " << endl;
//			cout << "P(c|h) is " << endl;
//			cout << c->transitionMatrix << endl;
//			cout << "P(h|p) is " << endl;
//			cout << h->transitionMatrix << endl;						
			// Set P(c|p) to P(h|p) * P(c|h)
//			cout << "P(c|p) = P(h|p) * P(c|h) is " << endl;
			c->transitionMatrix = h->transitionMatrix * c->transitionMatrix;
//			cout << c->transitionMatrix << endl;
			h->transitionMatrix = this->I4by4;
			// Remove (p,h) and (h,c)
//			cout << "Removing edge (p,h)" << endl;
			this->RemoveArc(p,h);
//			cout << "Removing edge (h,c)" << endl;
			this->RemoveArc(h,c);			
//			cout << "Adding edge (p,c)" << endl;			
			this->AddArc(p,c);
//			cout << "Edge (p,c) added" << endl;			
			tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeOne();
		}
		// Case 3. The root is a hidden vertex with out degree 1
		if (!this->root->observed and (this->root->outDegree == 1)) {
			if (debug) {
				cout << "Case 3. The root is a hidden vertex with out degree 1" << endl;
			}
			rootProb_orig = this->rootProbability;
			assert(this->root->children.size()==1);
			p = this->root;
			c = this->root->children[0];
			P = c->transitionMatrix;
			for (int y = 0; y < 4; y ++) {				
				rootProb_new[y] = 0;
				for (int x = 0; x < 4; x ++){
					rootProb_new[y] += rootProb_orig[x] * P(x,y);
				}
			}
			this->RemoveArc(p,c);
			this->root = c;
			this->rootProbability = rootProb_new;
			this->root->rootProbability = this->rootProbability;
			c->transitionMatrix = this->I4by4;
		}
		// Case 4. The root is an observed vertex
		if (this->root->observed) {
			if (debug) {
				cout << "Case 4. The root is an observed vertex" << endl;
			}
			tie (containsSingletonHiddenVertex, h_s) = this->CheckAndRetrieveSingletonHiddenVertex();
			assert(containsSingletonHiddenVertex);
			assert(h_s->children.size() == 0);
			p = h_s;
			c = this->root;
			childrenToSwap = c->children;
			for (SEM_vertex * child : childrenToSwap) {
				this->RemoveArc(c, child);
				this->AddArc(p, child);
			}
			this->root = p;
			this->root->rootProbability = this->rootProbability;
			this->AddArc(p,c);
			c->transitionMatrix = this->I4by4;
		}			
		// Case 5. p->o, o->c1, ..., o->ck
		// Check for non-leaf non-root observed vertex
		tie (containsMatchingVertex, matchingVertex) = this->CheckAndRetrieveObservedVertexThatIsNotALeafAndIsNotTheRoot();
		tie (containsSingletonHiddenVertex, h_s) = this->CheckAndRetrieveSingletonHiddenVertex();
		while (containsMatchingVertex and containsSingletonHiddenVertex) {
			if (debug) {
				cout << "Case 5. There is a non-leaf non-root observed vertex" << endl;
			}
			o = matchingVertex;
//			cout << o->name  << endl;
			// Swap children of o and h
			childrenToSwap = o->children;
			for (SEM_vertex * c: childrenToSwap){					
				this->RemoveArc(o,c);
				this->AddArc(h_s,c);
			}
			// Set parent of h to parent of o
			this->AddArc(o->parent,h_s);
			// Set P(h|p) to P(o|p)
			h_s->transitionMatrix = o->transitionMatrix;
			this->RemoveArc(o->parent,o);
			this->AddArc(h_s,o);
			// Set P(o|h) to Identity matrix I4by4
			o->transitionMatrix = this->I4by4;						
			tie (containsMatchingVertex, o) = this->CheckAndRetrieveObservedVertexThatIsNotALeafAndIsNotTheRoot();
			tie (containsSingletonHiddenVertex, h_s) = this->CheckAndRetrieveSingletonHiddenVertex();
		}
		// Case 6. p->h, h->c1, h->c2, ... h->ck
		// Check for a hidden vertex h with outdegree greater than two
		// and for a singleton hidden vertex h_s
		tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeGreaterThanTwo();
		tie (containsSingletonHiddenVertex, h_s) = this->CheckAndRetrieveSingletonHiddenVertex();			
		while (containsMatchingVertex and containsSingletonHiddenVertex) {			
			if (debug) {
				cout << "Case 6. There is a multifurcation" << endl;
			}
			childrenToSwap = h->children;
			sort(childrenToSwap.begin(),childrenToSwap.end(), [](SEM_vertex * u, SEM_vertex * v) {
				return u->id < v->id;
			});
			// Select children c_1 and c_2 are by sorting the children of h in ascending order of id
			// and selecting the first two chilren in the sorted list
			c_1 = childrenToSwap[0];
			c_2 = childrenToSwap[1];
			// Remove children c_1 and c_2 of h and set them as children of h_s
			this->RemoveArc(h,c_1);
			this->RemoveArc(h,c_2);
			this->AddArc(h_s,c_1);
			this->AddArc(h_s,c_2);
			// Set h as the parent of h_s
			this->AddArc(h,h_s);
			h_s->transitionMatrix = this->I4by4;
			tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeGreaterThanTwo();
			tie (containsSingletonHiddenVertex, h_s) = this->CheckAndRetrieveSingletonHiddenVertex();								
		}
		checkForCanonicalForm = IsTreeInCanonicalForm();
	}
//	cout << "Number of transformations equals " << numberOfTransformations << endl;
//	if (IsTreeInCanonicalForm()) {
//		cout << "Tree is in canonical form" << endl;
//	} else {
//		nonCanonicalRootedTreeFileName = "/home/pk/Projects/MSTBasedForests/data/trees/debugNonCanonicalRootedTree_after";
//		this->WriteRootedTreeAsEdgeList(nonCanonicalRootedTreeFileName);		
//	}
//	assert (IsTreeInCanonicalForm());
}

void SEM::ClearDirectedEdges() {
	// cout << "Resetting times visited " << endl;
	this->ResetTimesVisited();
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		v->parent = v;
		v->children.clear();		
		v->inDegree = 0;
		v->outDegree = 0;		
	}
}

void SEM::ClearUndirectedEdges() {
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		v->degree = 0;
		v->neighbors.clear();
	}
}

void SEM::ClearAllEdges() {
	this->ResetTimesVisited();
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		v->parent = v;
		v->children.clear();
		v->neighbors.clear();
		v->degree = 0;
		v->inDegree = 0;
		v->outDegree = 0;	
	}
}

void SEM::ResetTimesVisited() {
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;		
		v->timesVisited = 0;		
	}
}

array <float, 4> SEM::GetBaseComposition(SEM_vertex * v) {
	array <float, 4> baseCompositionArray;
	for (int dna = 0; dna < 4; dna ++) {
		baseCompositionArray[dna] = 0;
	}
	unsigned char dna_v;
	for (int site = 0; site < this->numberOfSitePatterns; site ++){
		dna_v = v->compressedSequence[site];
		baseCompositionArray[dna_v] += this->sitePatternWeights[site];
	}
	for (int dna = 0; dna < 4; dna ++) {
		baseCompositionArray[dna] /= sequenceLength;
	}
	return (baseCompositionArray);
}

array <float, 4> SEM::GetObservedCountsForVariable(SEM_vertex * v) {
	array <float, 4> observedCounts;
	for (int i = 0; i < 4; i++) {
		observedCounts[i] = 0;
 	}
	for (int site = 0; site < this->numberOfSitePatterns; site ++) {
		if (v->compressedSequence[site] < 4) { // FIX_AMB
			observedCounts[v->compressedSequence[site]] += this->sitePatternWeights[site];
		}		
	}
	return (observedCounts);
}


void SEM::ComputeMLEOfRootProbability() {
	this->rootProbability = GetBaseComposition(this->root);
	this->root->rootProbability = this->rootProbability;
}

void SEM::ComputeMLEOfTransitionMatrices() {
	SEM_vertex * c; SEM_vertex * p;
	bool debug = 0;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap){
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			c->transitionMatrix = this->GetTransitionMatrix(p,c);
			if (debug) {
				cout << "Estimated transition matrix is" << endl;
				cout << c->transitionMatrix << endl;
			}
		}
	}
	
}

void SEM::ComputeInitialEstimateOfModelParameters() {	
	bool debug = 0;
	this->rootProbability = GetBaseComposition(this->root);	
	this->root->rootProbability = this->rootProbability;
	if (debug) {
		cout << "Root probability is " << endl;
		for (int i = 0; i < 4; i++) {
			cout << this->rootProbability[i] << "\t";
		}
		cout << endl;
	}

	SEM_vertex * c; SEM_vertex * p;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap){
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			c->transitionMatrix = this->GetTransitionMatrix(p,c);		
			if (debug) {
				cout << "Transition matrix for " << p->name << " to " << c->name << " is " << endl;
				cout << c->transitionMatrix << endl;
			}			
		}
	}
	if (debug) {
		cout << "Transition matrices have been computed" << endl;
	}	
}


void SEM::ResetLogScalingFactors() {
	for (pair <int, SEM_vertex * > idPtrPair : * this->vertexMap){
		idPtrPair.second->logScalingFactors = 0;
	}
}

void SEM::ComputeMPEstimateOfAncestralSequences() {
	SEM_vertex * p;	
	map <SEM_vertex * , unsigned char> V;
	map <SEM_vertex * , vector<unsigned char>> VU;					
	map <unsigned char, int> dnaCount;
	unsigned char pos;
	unsigned char maxCount; unsigned char numberOfPossibleStates;
	if (this->root->compressedSequence.size() > 0){
		this->ClearAncestralSequences();
	}
	if (this->preOrderVerticesWithoutLeaves.size() == 0) {
		this->SetVerticesForPreOrderTraversalWithoutLeaves();
	}	
	//	Initialize sequences for ancestors
//	cout << "Length of compressed sequence for leaf 0 is ";
//	cout << this->leaves[0]->compressedSequence.size() << endl;
//	cout << "Number of site patterns is " << this->numberOfSitePatterns << endl;
	for (int site = 0; site < this->numberOfSitePatterns; site++) {
		V.clear();
		VU.clear();
	//	Compute V and VU for leaves
		for (SEM_vertex * c : this->leaves) {
//			cout << c->name << endl;
			assert (c->compressedSequence[site] < 4);			
			V.insert(make_pair(c,c->compressedSequence[site]));
//			cout << "Insert 1 successful" << endl;
			vector <unsigned char> vectorToAdd;
			vectorToAdd.push_back(c->compressedSequence[site]);			
			VU.insert(make_pair(c,vectorToAdd));
//			cout << "Insert 2 successful" << endl;
		}
	//	Set VU for ancestors
		for (SEM_vertex* c : this->preOrderVerticesWithoutLeaves) {
			vector <unsigned char> vectorToAdd;
			VU.insert(make_pair(c,vectorToAdd));
		}
		for (int p_ind = this->preOrderVerticesWithoutLeaves.size()-1; p_ind > -1; p_ind--) {
			p = preOrderVerticesWithoutLeaves[p_ind];			
			dnaCount.clear();
			for (unsigned char dna = 0; dna < 4; dna++) {
				dnaCount[dna] = 0;
			}
			for (SEM_vertex * c : p->children) {
				for (unsigned char dna: VU[c]) {
					dnaCount[dna] += 1;
				}
			}
			maxCount = 0;
			for (pair <unsigned char, int> dnaCountPair: dnaCount) {
				if (dnaCountPair.second > maxCount) {
					maxCount = dnaCountPair.second;
				}
			}			
			for (pair <unsigned char, int> dnaCountPair: dnaCount) { 
				if (dnaCountPair.second == maxCount) {
					VU[p].push_back(dnaCountPair.first);					
				}
			}			
		}
	// Set V for ancestors
		for (SEM_vertex * c : preOrderVerticesWithoutLeaves) {
			if (c->parent == c) {			
			// Set V for root
				if (VU[c].size()==1) {
//					cout << "Case 1a" << endl;
					V.insert(make_pair(c,VU[c][0]));
				} else {
//					cout << "Case 1b" << endl;
					numberOfPossibleStates = VU[c].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V.insert(make_pair(c,VU[c][pos]));
				}				
			} else {
//				cout << "Case 2" << endl;
				p = c->parent;
				if (find(VU[c].begin(),VU[c].end(),V[p])==VU[c].end()) {
					numberOfPossibleStates = VU[c].size();
					uniform_int_distribution <int> distribution(0,numberOfPossibleStates-1);
					pos = distribution(generator);
					V.insert(make_pair(c,VU[c][pos]));
				} else {
					V.insert(make_pair(c,V[p]));
				}				
			}
			// push states to compressedSequence	
			c->compressedSequence.push_back(V[c]);			
		}
	}		
}


void SEM::ComputeMAPEstimateOfAncestralSequences() {
	if (this->root->compressedSequence.size() > 0) {
		this->ClearAncestralSequences();
	}	
	this->logLikelihood = 0;
	float currentProbability;
	map <SEM_vertex*, array<float,4>> conditionalLikelihoodMap;
	array <float,4> conditionalLikelihood;
	float maxProbability;
	float stateWithMaxProbability;
	float partialLikelihood;
	float siteLikelihood;
	float largestConditionalLikelihood = 0;
	float currentProb;
	unsigned char dna_ind_p; unsigned char dna_ind_c;
	vector <SEM_vertex *> verticesToVisit;
	SEM_vertex * p;
	SEM_vertex * c;
	Matrix4f P;
	for (int site = 0 ; site < this->numberOfSitePatterns; site++){
		conditionalLikelihoodMap.clear();
		this->ResetLogScalingFactors();
		for (pair<SEM_vertex *,SEM_vertex *> edge : this->edgesForPostOrderTreeTraversal){
			tie (p, c) = edge;					
			P = c->transitionMatrix;
			p->logScalingFactors += c->logScalingFactors;				
			// Initialize conditional likelihood for leaves
			if (c->observed) {
				for (unsigned char dna_c = 0; dna_c < 4; dna_c ++){
					conditionalLikelihood[dna_c] = 0;
				}
				conditionalLikelihood[c->compressedSequence[site]] = 1;
				conditionalLikelihoodMap.insert(pair<SEM_vertex *,array<float,4>>(c,conditionalLikelihood));
			}
			// Initialize conditional likelihood for ancestors
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()){
				for (unsigned char dna_c = 0; dna_c < 4; dna_c++){
				conditionalLikelihood[dna_c] = 1;
				}
				conditionalLikelihoodMap.insert(pair<SEM_vertex *,array<float,4>>(p,conditionalLikelihood));
			}
			largestConditionalLikelihood = 0;
			for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
				partialLikelihood = 0;
				for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
//					if (P(dna_p,dna_c)*conditionalLikelihoodMap[c][dna_c] == 0 and P(dna_p,dna_c) > 0 and conditionalLikelihoodMap[c][dna_c] > 0) {
//						cout << "Numerical underflow in computing partial likelihood" << endl;
//						cout << "P(y|x) is " << P(dna_p,dna_c) << endl;
//						cout << "L(y) is " << conditionalLikelihoodMap[c][dna_c] << endl;								
//						cout << "2^-256 is " << 1.0/pow(2,256) << endl;
//					}
					partialLikelihood += P(dna_p,dna_c)*conditionalLikelihoodMap[c][dna_c];
				}
				conditionalLikelihoodMap[p][dna_p] *= partialLikelihood;
				if (conditionalLikelihoodMap[p][dna_p] > largestConditionalLikelihood) {
					largestConditionalLikelihood = conditionalLikelihoodMap[p][dna_p];
				}
			}
			if (largestConditionalLikelihood != 0){
				for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
					conditionalLikelihoodMap[p][dna_p] /= largestConditionalLikelihood;
				}
				p->logScalingFactors += log(largestConditionalLikelihood);
			} else {
				cout << "Largest conditional likelihood value is zero" << endl;
				cout << "Transition matrix P is" << endl;
				cout << P << endl;
				exit(-1);						
			}					
		}
		maxProbability = -1; stateWithMaxProbability = 10;	
		for (dna_ind_c = 0; dna_ind_c < 4; dna_ind_c ++) {
			currentProbability = this->rootProbability[dna_ind_c];
			currentProbability *= conditionalLikelihoodMap[this->root][dna_ind_c];
			if (currentProbability > maxProbability) {
				maxProbability = currentProbability;
				stateWithMaxProbability = dna_ind_c;
			}
		}
		if (stateWithMaxProbability > 3) {
			cout << maxProbability << "\tError in computing maximum a posterior estimate for ancestor vertex\n";
		} else {
			this->root->compressedSequence.push_back(stateWithMaxProbability);
		}
//		Compute MAP estimate for each ancestral sequence
		for (pair <SEM_vertex *,SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {			
			tie (p, c) = edge;
			P = c->transitionMatrix;
			if (!c->observed) {
				maxProbability = -1; stateWithMaxProbability = 10;
				dna_ind_p = p->compressedSequence[site];
				for (dna_ind_c = 0; dna_ind_c < 4; dna_ind_c ++){ 
					currentProbability = P(dna_ind_p,dna_ind_c);
					currentProbability *= conditionalLikelihoodMap[c][dna_ind_c];
					if (currentProbability > maxProbability) {
						maxProbability = currentProbability;
						stateWithMaxProbability = dna_ind_c;
					}
				}
				if (stateWithMaxProbability > 3) {
//					cout << "Error in computing maximum a posterior estimate for ancestor vertex";
				} else {
					c->compressedSequence.push_back(stateWithMaxProbability);
				}
			}
		}		
		siteLikelihood = 0; 							
		for (int dna = 0; dna < 4; dna++) {
			currentProb = this->rootProbability[dna]*conditionalLikelihoodMap[this->root][dna];
			siteLikelihood += currentProb;					
		}
		this->logLikelihood += (this->root->logScalingFactors + log(siteLikelihood)) * this->sitePatternWeights[site];				
	}
}

void SEM::ComputeMAPEstimateOfAncestralSequencesUsingHardEM() {
	if (this->root->compressedSequence.size() != (unsigned int) this->numberOfSitePatterns) {
		this->ComputeMPEstimateOfAncestralSequences();
	}	
	map <SEM_vertex*,Matrix4f> transitionMatrices;	
	map <SEM_vertex*,std::array<double,4>> conditionalLikelihoodMap;
	std::array <double,4> conditionalLikelihood;	
	double partialLikelihood;
	double siteLikelihood;
	double currentLogLikelihood = 0;
	double previousLogLikelihood = 0;
	double largestCondionalLikelihood = 0;
	int iter = 0;
	int maxIter = 10;	
	unsigned char dna_p; unsigned char dna_c;
	char maxProbState;
	float rowSum;
	double currentProb;
	double maxProb;	
	bool continueEM = 1;
	vector <SEM_vertex *> verticesToVisit;	
	SEM_vertex * p;
	SEM_vertex * c;
	Matrix4f P;
//	this->SetEdgesForPostOrderTreeTraversal();
	// Iterate till convergence of log likelihood
		while (continueEM and iter < maxIter) {
			iter += 1;			
			cout << "root sequence is " << endl;
			cout << EncodeAsDNA(this->root->compressedSequence) << endl;
			currentLogLikelihood = 0;
			// Estimate root probablity
			this->ComputeMLEOfRootProbability();		
			// Estimate transition matrices	
//			cout << "here 1" << endl;
			for (pair<int,SEM_vertex *> idPtrPair : *this->vertexMap) {
				c = idPtrPair.second;
				if (c->parent != c) {
					p = c->parent;
					P = ArrayXXf::Zero(4,4);
					for (int site = 0; site < this->numberOfSitePatterns; site++) {
						dna_p = p->compressedSequence[site];
						dna_c = c->compressedSequence[site];
						P(dna_p,dna_c) += this->sitePatternWeights[site];
					}
					for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
						rowSum = 0;
						for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
							rowSum += P(dna_p,dna_c);
						}
						for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
							 P(dna_p,dna_c) /= rowSum;
						}
					}
					c->transitionMatrix = P;
//					transitionMatrices.insert(pair<SEM_vertex*,Matrix4f>(c,P));
				}
			}
//			cout << "here 2" << endl;
			// Estimate ancestral sequences
			for (pair<int,SEM_vertex *> idPtrPair : (*this->vertexMap)) {
				c = idPtrPair.second;
				if (c->outDegree > 0) {
					c->compressedSequence.clear();
				}
			}		
			// Iterate over sites		
			for (int site = 0 ; site < this->numberOfSitePatterns; site++) {
				conditionalLikelihoodMap.clear();
				this->ResetLogScalingFactors();
//				cout << "site is " << site << endl;
				for (pair <SEM_vertex *,SEM_vertex *> edge : this->edgesForPostOrderTreeTraversal) {
					tie (p, c) = edge;					
					P = c->transitionMatrix;	
					p->logScalingFactors += c->logScalingFactors;				
					// Initialize conditional likelihood for leaves
					if (c->outDegree==0) {
						for (unsigned char dna_c = 0; dna_c < 4; dna_c ++) {
							conditionalLikelihood[dna_c] = 0;
						}
						conditionalLikelihood[c->compressedSequence[site]] = 1;
						conditionalLikelihoodMap.insert(pair <SEM_vertex *, array<double,4>>(c,conditionalLikelihood));
					}
					// Initialize conditional likelihood for ancestors
					if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()) {
						for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
						conditionalLikelihood[dna_c] = 1;
						}				
						conditionalLikelihoodMap.insert(pair <SEM_vertex *,array<double,4>>(p,conditionalLikelihood));					
					}		
					for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
						partialLikelihood = 0;
						for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
							partialLikelihood += P(dna_p,dna_c)*conditionalLikelihoodMap[c][dna_c];
						}
						conditionalLikelihoodMap[p][dna_p] *= partialLikelihood;
					}
					largestCondionalLikelihood = 0;
					for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
						if (conditionalLikelihoodMap[p][dna_p] > largestCondionalLikelihood) {
							largestCondionalLikelihood = conditionalLikelihoodMap[p][dna_p];
						}
					}
					if (largestCondionalLikelihood != 0) {
						for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
							conditionalLikelihoodMap[p][dna_p] /= largestCondionalLikelihood;
						}
						p->logScalingFactors += log(largestCondionalLikelihood);
					} else {
						cout << "Largest conditional likelihood value is zero" << endl;
						cout << "Transition matrix P is" << endl;
						cout << P << endl;
						exit(-1);						
					}					
				}
				maxProbState = -1;
				maxProb = 0;
				siteLikelihood = 0;
				for (int dna = 0; dna < 4; dna++) {
					currentProb = this->rootProbability[dna]*conditionalLikelihoodMap[this->root][dna];
					siteLikelihood += currentProb;
					if (currentProb > maxProb) {
						maxProb = currentProb;
						maxProbState = dna;		
					}
				}				
				currentLogLikelihood += (this->root->logScalingFactors + log(siteLikelihood)) * this->sitePatternWeights[site];
				if (maxProbState == -1) {
					cout << "check state estimation" << endl;					
				}
				this->root->compressedSequence.push_back(maxProbState);
				verticesToVisit.clear();			
				for (SEM_vertex * c: this->root->children) {
					if (c->outDegree > 0) {
						verticesToVisit.push_back(c);
					}
				}				
				for (pair<SEM_vertex *, SEM_vertex*> edge : this->edgesForPreOrderTreeTraversal) {				
					tie (p, c) = edge;
					if (c->outDegree > 0) {
						P = transitionMatrices[c];
						dna_p = p->compressedSequence[site];
						maxProbState = -1;
						maxProb = 0;
						for (int dna_c = 0; dna_c < 4; dna_c++) {
							currentProb = P(dna_p,dna_c)*conditionalLikelihoodMap[c][dna_c];
							if (currentProb > maxProb) {
								maxProb = currentProb;
								maxProbState = dna_c;
							}
						}
						if (maxProbState == -1) {
							cout << "check state estimation" << endl;
						}
						c->compressedSequence.push_back(maxProbState);					
					}
				}
			}
			if (iter < 2 or currentLogLikelihood > previousLogLikelihood or abs(currentLogLikelihood-previousLogLikelihood) > 0.001) {
				continueEM = 1;
				previousLogLikelihood = currentLogLikelihood;
			} else {
				continueEM = 0;
			}
		}
		this->ResetLogScalingFactors();
		this->logLikelihood = currentLogLikelihood;
}

void SEM::ComputePosteriorProbabilitiesUsingMAPEstimates() {
	this->posteriorProbabilityForVertexPair.clear();
	Matrix4f P;
	SEM_vertex * u;
	SEM_vertex * v;
	float sum;
	unsigned char dna_u; unsigned char dna_v;	
	for (unsigned int u_id = 0; u_id < this->vertexMap->size()-1; u_id ++) {
		u = (*this->vertexMap)[u_id];		
		// Posterior probability for vertex u
		u->posteriorProbability = this->GetBaseComposition(u);
		// Posterior probabilies for vertex pair (u,v)
		for (unsigned int v_id = u_id + 1 ; v_id < this->vertexMap->size()-1; v_id ++) {			
			v = (*this->vertexMap)[v_id];
			P = ArrayXXf::Zero(4,4);
			for (int site = 0; site < this->numberOfSitePatterns; site++ ) {		
				dna_u = u->compressedSequence[site];
				dna_v = v->compressedSequence[site];
				P(dna_u,dna_v) += this->sitePatternWeights[site];
			}
			sum = 0;
			for (dna_u = 0; dna_u < 4; dna_u ++) {				
				for (dna_v = 0; dna_v < 4; dna_v ++) {
					sum += P(dna_u, dna_v);
				}				
			}
			for (dna_u = 0; dna_u < 4; dna_u ++) {				
				for (dna_v = 0; dna_v < 4; dna_v ++) {
					P(dna_u, dna_v) /= sum;
				}				
			}			
			this->posteriorProbabilityForVertexPair.insert(make_pair(make_pair(u,v),P));
		}
		
	}	
}

float SEM::GetExpectedMutualInformation(SEM_vertex * x, SEM_vertex* y) {
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (x->id < y->id) {
		vertexPair = pair <SEM_vertex *, SEM_vertex *>(x,y);
	} else {
		vertexPair = pair <SEM_vertex *, SEM_vertex *>(y,x);
	}
	
	Matrix4f P = this->posteriorProbabilityForVertexPair[vertexPair];
//	cout << "Joint probability for vertex pair " << x->name << "\t" << y->name << " is " << endl;
//	cout << P << endl;
	std::array <float, 4> P_x;
	std::array <float, 4> P_y;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		P_x[dna_x] = 0;
		P_y[dna_x] = 0;
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			P_x[dna_x] += P(dna_x,dna_y);
			P_y[dna_x] += P(dna_y,dna_x);
		}		
	}
//	cout << "P_x is ";
//	for (int i = 0; i < 4; i++) {
//		cout << P_x[i] << "\t";
//	}
//	cout << endl << "P_y is ";
//	for (int i = 0; i < 4; i++) {
//		cout << P_y[i] << "\t";
//	}
//	cout << endl;
	float mutualInformation = 0;
//	float inc;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			if (P(dna_x,dna_y) > 0) {
//				inc = P(dna_x,dna_y) * log(P(dna_x,dna_y)/(P_x[dna_x] * P_y[dna_y]));
//				cout << "Incrementing mutual information by " << inc << endl; 
				mutualInformation += P(dna_x,dna_y) * log(P(dna_x,dna_y)/(P_x[dna_x] * P_y[dna_y]));
			}
		}
	}
//	cout << "P_XY is " << endl << P << endl;
//	cout << "mutual information is " << mutualInformation << endl;
	return (mutualInformation);
}

void SEM::ComputeChowLiuTree() {
	using namespace boost;	
	this->ClearAllEdges();
	int numberOfVertices = this->vertexMap->size();
	for (int i = 0; i < numberOfVertices; i++) {
		assert ((*this->vertexMap).find(i) != (*this->vertexMap).end());
	}
	const int numberOfEdges = numberOfVertices * (numberOfVertices-1)/2;	
	float maxMutualInformation = 0;
	float mutualInformation;
	float * shiftedNegMutualInformation;
	shiftedNegMutualInformation = new float [numberOfEdges];		
	SEM_vertex * u; SEM_vertex * v;	
	int edgeIndex = 0;
	for (int i=0; i<numberOfVertices; i++) {
		u = (*this->vertexMap)[i];
		for (int j=i+1; j<numberOfVertices; j++) {
			v = (*this->vertexMap)[j];
			mutualInformation = this->GetExpectedMutualInformation(u,v);
			shiftedNegMutualInformation[edgeIndex] = -1 * mutualInformation;
			if (mutualInformation > maxMutualInformation) {
				maxMutualInformation = mutualInformation;
			}
			edgeIndex += 1;
		}
	}
		
	// Compute shifted mutual information as follows
	// shifted_negative_MI = max ({MI}) - MI + 0.00001
	// 0.00001 is added to ensure that smallest shifted_negative_MI is greater than 0
	maxMutualInformation += 0.00001;
	for (int i = 0; i < numberOfEdges; i++) {
		shiftedNegMutualInformation[i] += maxMutualInformation;
	}
	
	typedef pair <int, int> E;

	E * edges;
	edges = new E [numberOfEdges];
	edgeIndex = 0;
	for (int i=0; i<numberOfVertices; i++) {
		for (int j=i+1; j<numberOfVertices; j++) {
			edges[edgeIndex] = E(i,j);
			edgeIndex += 1;
		}
	}
	
	typedef adjacency_list <vecS, vecS, undirectedS, property <vertex_distance_t, int>, property <edge_weight_t, float> > Graph;
	Graph g(edges, edges + numberOfEdges, shiftedNegMutualInformation, numberOfVertices);
	vector < graph_traits < Graph >::vertex_descriptor >  p(num_vertices(g));
	prim_minimum_spanning_tree(g, &p[0]);	
	
	for (size_t i = 0; i != p.size(); i++) {
		if (p[i] != i) {
			u = (*this->vertexMap)[i];
			v = (*this->vertexMap)[p[i]];
			u->AddNeighbor(v);
			v->AddNeighbor(u);
			if (i < p[i]){
				edgeIndex = this->GetEdgeIndex(i,p[i],numberOfVertices);
			} else {
				edgeIndex = this->GetEdgeIndex(p[i],i,numberOfVertices);
			}
//			if (u->name == "l_1" or v->name == "l_1") {
//				cout << u->name << "\t" << v->name << "\t" << shiftedNegMutualInformation[edgeIndex] << endl;
//				if (u->id < v->id) {
//					cout << this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(u,v)] << endl;
//				} else {
//					cout << this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(v,u)] << endl;
//				}				
//			}
		}
	}	
	delete[] edges;
	delete[] shiftedNegMutualInformation;
}

void SEM::OptimizeTopology() {
	// ClearEdges
	this->ClearDirectedEdges();
	// Reset edges using Chow-Liu tree
	this->ComputeChowLiuTree();
	// Select vertex for rooting
	// Modify topology such that tree is bifurcating
	// Swap edgs of vertex with in-degree 0 and vertex with id -1;
}

void SEM::SetEdgesForTreeTraversalOperations() {
	this->SetEdgesForPreOrderTraversal();
	this->SetEdgesForPostOrderTraversal();
	this->SetVerticesForPreOrderTraversalWithoutLeaves();	
	this->SetLeaves();
}

void SEM::SetEdgesForPreOrderTraversal() {
	this->edgesForPreOrderTreeTraversal.clear();
	vector <SEM_vertex*> verticesToVisit;	
	SEM_vertex * p;	
	verticesToVisit.push_back(this->root);
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		p = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex* c : p->children){
			this->edgesForPreOrderTreeTraversal.push_back(pair<SEM_vertex*, SEM_vertex*>(p,c));
			verticesToVisit.push_back(c);
			numberOfVerticesToVisit += 1;
		}
	}
}

void SEM::SetLeaves() {
	this->leaves.clear();
	for (pair<int, SEM_vertex*> idPtrPair : * this->vertexMap){
		if (idPtrPair.second->outDegree == 0) {
			this->leaves.push_back(idPtrPair.second);
		}
	}
}

void SEM::SetEdgesForPostOrderTraversal() {	
	vector <SEM_vertex*> verticesToVisit;	
	SEM_vertex* c;
	SEM_vertex* p;	
	for (pair<int,SEM_vertex*> idPtrPair : *this->vertexMap){
		idPtrPair.second->timesVisited = 0;		
	}	
	if (this->leaves.size()== 0) {
		this->SetLeaves();
	}
	this->edgesForPostOrderTreeTraversal.clear();	
	verticesToVisit = this->leaves;
	int numberOfVerticesToVisit = verticesToVisit.size();	
	while (numberOfVerticesToVisit > 0) {
		c = verticesToVisit[numberOfVerticesToVisit -1];
		verticesToVisit.pop_back();
		numberOfVerticesToVisit -= 1;
		if (c != c->parent) {
			p = c->parent;
			this->edgesForPostOrderTreeTraversal.push_back(pair<SEM_vertex*, SEM_vertex*>(p,c));
			p->timesVisited += 1;
			if (p->timesVisited == p->outDegree) {
				verticesToVisit.push_back(p);
				numberOfVerticesToVisit += 1;
			}
		}
	}
}

void SEM::SetVerticesForPreOrderTraversalWithoutLeaves() {
	this->preOrderVerticesWithoutLeaves.clear();
	for (pair <SEM_vertex*, SEM_vertex*> edge : this->edgesForPreOrderTreeTraversal) {
		if (find(this->preOrderVerticesWithoutLeaves.begin(),this->preOrderVerticesWithoutLeaves.end(),edge.first) == this->preOrderVerticesWithoutLeaves.end()){
			this->preOrderVerticesWithoutLeaves.push_back(edge.first);
		}
	}
}

void SEM::RootedTreeAlongAnEdgeIncidentToCentralVertex() {	
//	cout << "here 1" << endl;	
	// Identify a central vertex
	vector <SEM_vertex*> verticesToVisit;
	vector <SEM_vertex*> verticesVisited;
	SEM_vertex * u; SEM_vertex * v;
	int n_ind; int u_ind;
	for (pair <int, SEM_vertex *> idPtrPair : *this->vertexMap) {
		v = idPtrPair.second;
		v->timesVisited = 0;
		if (v->observed) {
			verticesToVisit.push_back(v);
		}
	}
//	cout << "here 2" << endl;
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		v = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(v);		
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex* n: v->neighbors) {
			if (find(verticesVisited.begin(),verticesVisited.end(),n)==verticesVisited.end()) {
				n->timesVisited += 1;
				if ((n->degree - n->timesVisited) == 1) {
					verticesToVisit.push_back(n);
					numberOfVerticesToVisit += 1;
				}
			} else {				
				n_ind = find(verticesVisited.begin(),verticesVisited.end(),n) - verticesVisited.begin();
				verticesVisited.erase(verticesVisited.begin()+n_ind);
			}
		}
	}
//	cout << "here 3" << endl;
//	cout << "v id is " << v->id << endl;
	// v is a central vertex	
	// Root tree at a randomly selected neighbor u of v
	uniform_int_distribution <int> distribution(0,v->neighbors.size()-1);
//	cout << "here as well" << endl;
	u_ind = distribution(generator);
	u = v->neighbors[u_ind];
//	cout << "here 4" << endl;
	this->root->AddChild(u);
	this->root->AddChild(v);
	u->AddParent(this->root);
	v->AddParent(this->root);
//	cout << "here 5" << endl;
//	cout << "rooting at edge" << u->id << "\t" << v->id << endl;
	verticesToVisit.clear();
	verticesVisited.clear();
	verticesToVisit.push_back(u);
	verticesToVisit.push_back(v);
	verticesVisited.push_back(u);
	verticesVisited.push_back(v);
	numberOfVerticesToVisit = verticesToVisit.size() - 1;
//	cout << "here 6" << endl;
	while (numberOfVerticesToVisit > 0) {
		v = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(v);
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex* n: v->neighbors) {
			if (find(verticesVisited.begin(),verticesVisited.end(),n)==verticesVisited.end()) {
				verticesToVisit.push_back(n);
				numberOfVerticesToVisit += 1;
				v->AddChild(n);
				n->AddParent(v);
			}
		}
	}
//	cout << "here 7" << endl;
	this->SetLeaves();	
	this->SetEdgesForPreOrderTraversal();
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
	this->SetEdgesForPostOrderTraversal();
//	cout << "here 6" << endl;
}

bool SEM::IsNumberOfNonSingletonComponentsGreaterThanZero() {	
	bool valueToReturn;
	if (this->indsOfVerticesOfInterest.size() > 0) {
		valueToReturn = 1;
	} else {
		valueToReturn = 0;
	}
	return (valueToReturn);
}

void SEM::SelectIndsOfVerticesOfInterestAndEdgesOfInterest() {
	this->SuppressRoot();
	vector <SEM_vertex *> verticesToVisit;
	vector <SEM_vertex *> verticesVisited;
	int n_ind; SEM_vertex * v;
	pair <int, int> edgeToAdd;
	this->indsOfVerticesOfInterest.clear();
	this->indsOfVerticesToKeepInMST.clear();
	this->edgesOfInterest_ind.clear();
	bool vertex_n_NotVisited;
//	cout << "Selecting edges of interest" << endl;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		v->timesVisited = 0;
		if (v->observed and v->id < this->numberOfVerticesInSubtree) {
			verticesToVisit.push_back(v);
		}
	}
	int numberOfVerticesToVisit = verticesToVisit.size();
	while (numberOfVerticesToVisit > 0) {
		v = verticesToVisit[numberOfVerticesToVisit-1];
		verticesToVisit.pop_back();
		verticesVisited.push_back(v);		
		numberOfVerticesToVisit -= 1;
		for (SEM_vertex * n: v->neighbors) {
			vertex_n_NotVisited = find(verticesVisited.begin(),verticesVisited.end(),n) == verticesVisited.end();
			if (vertex_n_NotVisited) {
				n->timesVisited += 1;
				if ((n->degree - n->timesVisited) == 1) {
					verticesToVisit.push_back(n);
					numberOfVerticesToVisit +=1;
				}
			} else {
				edgesOfInterest_ind.push_back(pair<int,int>(n->id,v->id));
//					cout << v->name << "\t" << n->name << endl;
//					cout << "\tLocal id\t" << v->id;
//					cout << "\tGlobal id\t" << v->global_id << endl;
				if (n->observed) {
					this->idsOfVerticesToRemove.push_back(n->global_id);
				}
				n_ind = find(verticesVisited.begin(),verticesVisited.end(),n) - verticesVisited.begin();
				verticesVisited.erase(verticesVisited.begin() + n_ind);					
			}
		}
	}
	for (SEM_vertex * v: verticesVisited) {	
		if (!v->observed) {
			this->indsOfVerticesOfInterest.push_back(v->id);
		} else {
			this->idsOfVerticesToKeepInMST.push_back(v->global_id);
		}
	}	
}

void SEM::RenameHiddenVerticesInEdgesOfInterestAndSetIdsOfVerticesOfInterest() {		
	vector <SEM_vertex *> vertices;	
	for (pair <int, int> edge: this->edgesOfInterest_ind) {		
		vertices.push_back((*this->vertexMap)[edge.first]);
		vertices.push_back((*this->vertexMap)[edge.second]);
		for (SEM_vertex * v : vertices) {
			if (v->global_id < 0) {				
				v->global_id = this->largestIdOfVertexInMST;
				this->largestIdOfVertexInMST += 1;
				v->name = "h_" + to_string(v->global_id - this->numberOfInputSequences +1);
			}
		}		
	}
	this->idsOfVerticesOfInterest.clear();
	SEM_vertex * v;	
	for (int v_id : this->indsOfVerticesOfInterest) {
		v = (*this->vertexMap)[v_id];
		this->idsOfVerticesOfInterest.push_back(v->global_id);
	}
}

void SEM::SetIdsOfExternalVertices() {	
	this->idsOfExternalVertices.clear();
	SEM_vertex * v;
	for (int i = this->numberOfVerticesInSubtree; i < this->numberOfObservedVertices; i++) {		
		v = (*this->vertexMap)[i];
		this->idsOfExternalVertices.push_back(v->global_id);
	}
}

void SEM::SetInfoForVerticesToAddToMST(){
	this->idAndNameAndSeqTuple.clear();	
	vector <unsigned char> fullSeq;
	SEM_vertex * v;	
	for (int i : this->indsOfVerticesOfInterest){
		v = (*this->vertexMap)[i];
		fullSeq = DecompressSequence(&(v->compressedSequence),&(this->sitePatternRepetitions));
		this->idAndNameAndSeqTuple.push_back(make_tuple(v->global_id,v->name,fullSeq));
	}	
}

void SEM::AddSitePatternWeights(vector <int> sitePatternWeightsToAdd) {
	this->sitePatternWeights = sitePatternWeightsToAdd;
	this->numberOfSitePatterns = this->sitePatternWeights.size();
	this->sequenceLength = 0;
	for (int sitePatternWeight : this->sitePatternWeights) {
		this->sequenceLength += sitePatternWeight;
	}
}

void SEM::AddSitePatternRepeats(vector <vector <int> > sitePatternRepetitionsToAdd) {
	this->sitePatternRepetitions = sitePatternRepetitionsToAdd;
}

void SEM::SetNumberOfVerticesInSubtree(int numberOfVerticesToSet) {
	this->numberOfVerticesInSubtree = numberOfVerticesToSet;
}

void SEM::AddNames(vector <string> namesToAdd) {
	if (this->numberOfObservedVertices == 0) {
		this->numberOfObservedVertices = namesToAdd.size();
	}	
	for (int i = 0; i < this->numberOfObservedVertices; i++) {
		(*this->vertexMap)[i]->name = namesToAdd[i];
		this->nameToIdMap.insert(make_pair(namesToAdd[i],i));
	}
	this->externalVertex = (*this->vertexMap)[this->numberOfObservedVertices-1];
}

void SEM::AddGlobalIds(vector <int> idsToAdd) {
	if (this->numberOfObservedVertices == 0) {
		this->numberOfObservedVertices = idsToAdd.size();
	}
	SEM_vertex * v;
	for (int i = 0; i < this->numberOfObservedVertices; i++) {
		v = (*this->vertexMap)[i];
		v->global_id = idsToAdd[i];
//		cout << "Name\t" << v->name;
//		cout << "\tLocal id\t" << v->id;
//		cout << "\tGlobal id\t" << v->global_id << endl;
	}
}

void SEM::AddSequences(vector <vector <unsigned char>> sequencesToAdd) {
	this->numberOfObservedVertices = sequencesToAdd.size();
	this->h_ind = this->numberOfObservedVertices;
	for (int i = 0 ; i < this->numberOfObservedVertices; i++) {
		SEM_vertex * v = new SEM_vertex(i,sequencesToAdd[i]);
		v->observed = 1;
		this->vertexMap->insert(make_pair(i,v));
	}	
}

void SEM::AddRootVertex() {
	int n = this->numberOfObservedVertices;
	vector <unsigned char> emptySequence;	
	this->root = new SEM_vertex (-1,emptySequence);
	this->root->name = "h_root";	
	this->root->id = ( 2 * n ) - 2;
	this->vertexMap->insert(pair<int,SEM_vertex*>((( 2 * n ) - 2 ),this->root));
	this->nameToIdMap.insert(make_pair(this->root->name,this->root->id));
}

void SEM::AddWeightedEdges(vector < tuple <string,string,float> > weightedEdgesToAdd) {
	SEM_vertex * u; SEM_vertex * v;
	string u_name; string v_name; float t;
	vector <unsigned char> emptySequence;
	for (tuple <string, string, float> weightedEdge : weightedEdgesToAdd) {
		tie (u_name, v_name, t) = weightedEdge;		
		if (this->ContainsVertex(u_name)) {
			u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		} else {
			if (!this->ContainsVertex(v_name)) {
				cout << "Adding edge " << u_name << "\t" << v_name << endl;
			}
			assert(this->ContainsVertex(v_name));
//			cout << "Contains " << v_name << endl;
			u = new SEM_vertex(this->h_ind,emptySequence);
			u->name = u_name;
			u->id = this->h_ind;
			this->vertexMap->insert(pair<int,SEM_vertex*>(u->id,u));
			this->nameToIdMap.insert(make_pair(u->name,u->id));			
			this->h_ind += 1;
			assert(this->ContainsVertex(u_name));
		}
		
		if (this->ContainsVertex(v_name)) {
			v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		} else {
			if (!this->ContainsVertex(u_name)) {
				cout << "Adding edge " << u_name << "\t" << v_name << endl;
			}
			assert(this->ContainsVertex(u_name));
//			cout << "Contains " << u_name << endl;
			v = new SEM_vertex(this->h_ind,emptySequence);
			v->name = v_name;
			v->id = this->h_ind;
			this->vertexMap->insert(pair<int,SEM_vertex*>(v->id,v));
			this->nameToIdMap.insert(make_pair(v->name,v->id));
			this->h_ind += 1;
			assert(this->ContainsVertex(v_name));
		}
//		cout << "Adding neighbors" << endl;
		u->AddNeighbor(v);
		v->AddNeighbor(u);		
		if (u->id < v->id) {
			this->edgeLengths.insert(make_pair(make_pair(u,v),t));			
		} else {
			this->edgeLengths.insert(make_pair(make_pair(v,u),t));
		}
//		cout << "Adding edge lengths" << endl;		
	}
}

void SEM::AddEdgeLogLikelihoods(vector<tuple<string,string,float>> edgeLogLikelihoodsToAdd) {
	SEM_vertex * u; SEM_vertex * v; float edgeLogLikelihood;
	string u_name; string v_name;
	pair<SEM_vertex *, SEM_vertex *> vertexPair;
	for (tuple<string,string,float> edgeLogLikelihoodTuple : edgeLogLikelihoodsToAdd) {
		tie (u_name, v_name, edgeLogLikelihood) = edgeLogLikelihoodTuple;
		u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		vertexPair = pair <SEM_vertex *, SEM_vertex *> (u,v);
		this->edgeLogLikelihoodsMap.insert(pair<pair <SEM_vertex *, SEM_vertex *>,float>(vertexPair, edgeLogLikelihood));
	}	
}

void SEM::AddVertexLogLikelihoods(map<string,float> vertexLogLikelihoodsMapToAdd) {
	string v_name; SEM_vertex * v; float vertexLogLikelihood;
	for (pair<string,float> vNameAndLogLik : vertexLogLikelihoodsMapToAdd) {
		tie (v_name, vertexLogLikelihood) = vNameAndLogLik;
		v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		v->vertexLogLikelihood = vertexLogLikelihood;
	}
}

bool SEM::ContainsVertex(string v_name) {	
	if (this->nameToIdMap.find(v_name) == this->nameToIdMap.end()) {		
		return (0);
	} else {
		return (1);
	}
}

//void SEM::AddCompressedSequencesAndNames(map<string,vector<unsigned char>> sequencesList, vector <vector <int>> sitePatternRepeats) {
//	string v_name;
//	vector <unsigned char> compressedSequence;
//	vector <unsigned char> fullSequence;	
//	for (pair <string, vector <unsigned char>> nameAndSeq : sequencesList) {
//		v_name = nameAndSeq.first;		    
//		if (!this->ContainsVertex(v_name)) {
//			compressedSequence = nameAndSeq.second;
//			fullSequence = DecompressSequence(&compressedSequence, &sitePatternRepeats);
////			this->AddVertex(v_name, fullSequence);
////			this->ComputeVertexLoglikelihood(this->vertices[v_name]);
//		}
//	}
//}

float SEM::ComputeDistance(int v_i, int v_j) {
	vector <unsigned char> seq_i; vector <unsigned char> seq_j;	
	seq_i = (*this->vertexMap)[v_i]->compressedSequence;
	seq_j = (*this->vertexMap)[v_j]->compressedSequence;
	float sequence_length = 0;
	for (int site = 0; site < numberOfSitePatterns; site++) {
		sequence_length += float(this->sitePatternWeights[site]);
	}
	// logDet_distance
	float distance;
	if (flag_logDet) {
		float logDet;
		float det_of_count_frequency_matrix;
		array <float, 4> f_i;
		array <float, 4> f_j;
		Matrix4f F = ArrayXXf::Zero(4,4);
		f_i[0] = 0.0;
		f_i[1] = 0.0;
		f_i[2] = 0.0;
		f_i[3] = 0.0;
		f_j[0] = 0.0;
		f_j[1] = 0.0;
		f_j[2] = 0.0;
		f_j[3] = 0.0;
		for (int site = 0; site < numberOfSitePatterns; site++) {
			// cout << seq_i[site] << "\t" << seq_j[site] << endl;
			F(seq_i[site],seq_j[site]) += float(this->sitePatternWeights[site])/sequence_length;
			f_i[seq_i[site]] += float(this->sitePatternWeights[site])/sequence_length;
			f_j[seq_j[site]] += float(this->sitePatternWeights[site])/sequence_length;
		}
		det_of_count_frequency_matrix = F.determinant();
		if (det_of_count_frequency_matrix <= 0) {
			cout << "count frequency matrix is " << endl;
			cout << F << endl;
		}
		assert (det_of_count_frequency_matrix > 0);
		distance = log(det_of_count_frequency_matrix);
		for (int dna = 0; dna < 4; dna ++) {
			distance -= 0.5 * log(f_i[dna]);
			distance -= 0.5 * log(f_j[dna]);
		}
		distance *= -0.25;
	} else if (flag_Hamming) {
		distance = 0;
		for (int site = 0; site < numberOfSitePatterns; site++) {
			if (seq_i[site] != seq_j[site]) {
			distance += float(this->sitePatternWeights[site])/sequence_length;
			}						
		}				
	}
	return (distance);
}

void SEM::ComputeNJTree() {
	
	// cout << "Computing NJ tree using LogDet distances" << endl;
	// Compute distances	
	map <pair <int,int>, float> distanceMap;
	map <int,float> R;
	vector <int> vertexIndsForIterating;
	vector <unsigned char> emptySequence;
	unsigned int n = this->numberOfObservedVertices;			
	float distance;		
	for (unsigned int i = 0; i < n; i++) {
		R[i] = 0.0;
		vertexIndsForIterating.push_back(i);			
		for (unsigned int j = i+1; j < n; j++) {	
			distance = this->ComputeDistance(i,j);
			if (this->verbose) {
				cout << "Distance measure for " << (*this->vertexMap)[i]->name << "\t" << (*this->vertexMap)[j]->name << "\t"<<this->distance_measure_for_NJ << "\t" << distance << endl;
			}			
			distanceMap[pair<int,int>(i,j)] = distance;
			R[i] += distance;
			R[j] += distance;
		}
	}

	// NJ algorithm
	for (unsigned int i = 0; i < n; i++){R[i] /= (n-2);}
	float neighborDist;
	float neighborDist_current;
	float newDist;
	int i; int j; int k;
	int i_selected; int j_selected;
	int h = n;
	
	while (R.size() > 3) {
		neighborDist = float(this->sequenceLength+1);
        for (unsigned int i_ind = 0; i_ind < n; i_ind++) {
			for (unsigned int j_ind = i_ind+1; j_ind < n; j_ind++) {
				i = vertexIndsForIterating[i_ind];
				j = vertexIndsForIterating[j_ind];
				neighborDist_current = distanceMap[pair<int,int>(i,j)]-R[i]-R[j];
				if (neighborDist_current < neighborDist){
					neighborDist = neighborDist_current;
                    i_selected = i;
                    j_selected = j;
				}
			}
		}
		vertexIndsForIterating.erase(remove(vertexIndsForIterating.begin(),vertexIndsForIterating.end(),i_selected),vertexIndsForIterating.end());
		vertexIndsForIterating.erase(remove(vertexIndsForIterating.begin(),vertexIndsForIterating.end(),j_selected),vertexIndsForIterating.end());		
		vertexIndsForIterating.push_back(h);
		SEM_vertex* h_ptr = new SEM_vertex (h,emptySequence);
		h_ptr->name = "h_" + to_string(this->h_ind);
		this->h_ind += 1;
		this->vertexMap->insert(pair<int,SEM_vertex*>(h,h_ptr));
		(*this->vertexMap)[i_selected]->AddNeighbor((*this->vertexMap)[h]);
		(*this->vertexMap)[j_selected]->AddNeighbor((*this->vertexMap)[h]);
		(*this->vertexMap)[h]->AddNeighbor((*this->vertexMap)[i_selected]);
		(*this->vertexMap)[h]->AddNeighbor((*this->vertexMap)[j_selected]);

        R.erase(i_selected);
        R.erase(j_selected);
        R[h] = 0.0;
		n -= 1;
		for (unsigned int k_ind = 0; k_ind < n-1; k_ind++) {
			k = vertexIndsForIterating[k_ind];
			if (k < i_selected) {
				newDist = distanceMap[pair<int,int>(k,i_selected)] + distanceMap[pair<int,int>(k,j_selected)];
                newDist -= distanceMap[pair<int,int>(i_selected,j_selected)];
                newDist *= 0.5;
                R[k] = float(R[k]*(n-1)-distanceMap[pair<int,int>(k,i_selected)]-distanceMap[pair<int,int>(k,j_selected)] + newDist)/float(n-2);
                distanceMap.erase(pair<int,int>(k,i_selected));
				distanceMap.erase(pair<int,int>(k,j_selected));               
			} else if (j_selected < k) {
				newDist = distanceMap[pair<int,int>(i_selected,k)] + distanceMap[pair<int,int>(j_selected,k)];
                newDist -= distanceMap[pair<int,int>(i_selected,j_selected)];
                newDist *= 0.5;
                R[k] = float(R[k]*(n-1)-distanceMap[pair<int,int>(i_selected,k)]-distanceMap[pair<int,int>(j_selected,k)] + newDist)/float(n-2);
                distanceMap.erase(pair<int,int>(i_selected,k));
                distanceMap.erase(pair<int,int>(j_selected,k));
			} else {
			    newDist = distanceMap[pair<int,int>(i_selected,k)] + distanceMap[pair<int,int>(k,j_selected)];
                newDist -= distanceMap[pair<int,int>(i_selected,j_selected)];
                newDist *= 0.5;
                R[k] = float(R[k]*(n-1)-distanceMap[pair<int,int>(i_selected,k)]-distanceMap[pair<int,int>(k,j_selected)] + newDist)/float(n-2);
                distanceMap.erase(pair<int,int>(i_selected,k));
                distanceMap.erase(pair<int,int>(k,j_selected));
			}            
			distanceMap[pair<int,int>(k,h)] = newDist;
            R[h] += newDist;
		}
        R[h] /= float(n-2);
		h += 1;
        distanceMap.erase(pair<int,int>(i_selected,j_selected));
	}
	SEM_vertex* h_ptr = new SEM_vertex (h,emptySequence);
	h_ptr->name = "h_" + to_string(this->h_ind);
	this->h_ind += 1;
	this->vertexMap->insert(pair<int,SEM_vertex*>(h,h_ptr));
	for (int v:vertexIndsForIterating) {
		(*this->vertexMap)[v]->AddNeighbor((*this->vertexMap)[h]);
		(*this->vertexMap)[h]->AddNeighbor((*this->vertexMap)[v]);
	}
}

// void SEM::ComputeNJTreeUsingHammingDistances() {
// 	map <pair <int,int>, float> distanceMap;
// 	map <int,float> R;
// 	vector <int> vertexIndsForIterating;
// 	vector <unsigned char> emptySequence;
// 	unsigned int n = this->numberOfObservedVertices;
// 	for (unsigned int i = 0; i < n; i++) {
// 		R[i] = 0.0;
// 		vertexIndsForIterating.push_back(i);
// 	}
// 	float distance;
// 	vector <unsigned char> seq_i; vector <unsigned char> seq_j;
// 	for (unsigned int i = 0; i < n; i++) {
// 		seq_i = (*this->vertexMap)[i]->compressedSequence;
// 		for (unsigned int j = i+1; j < n; j++) {
// 			distance = 0.0;
// 			seq_j = (*this->vertexMap)[j]->compressedSequence;
// 			for (int site=0; site < numberOfSitePatterns; site++) {
// 				if (seq_i[site] != seq_j[site]) {
// 					distance += float(this->sitePatternWeights[site]);
// 				}
// 			}
// 			distanceMap[pair<int,int>(i,j)] = distance;
// 			R[i] += distance;
// 			R[j] += distance;
// 		}
// 	}
// 	for (unsigned int i = 0; i < n; i++){R[i] /= (n-2);}
// 	float neighborDist;
// 	float neighborDist_current;
// 	float newDist;
// 	int i; int j; int k;
// 	int i_selected; int j_selected;
// 	int h = n;
	
// 	while (R.size() > 3) {
// 		neighborDist = float(this->sequenceLength+1);
//         for (unsigned int i_ind = 0; i_ind < n; i_ind++) {
// 			for (unsigned int j_ind = i_ind+1; j_ind < n; j_ind++) {
// 				i = vertexIndsForIterating[i_ind];
// 				j = vertexIndsForIterating[j_ind];
// 				neighborDist_current = distanceMap[pair<int,int>(i,j)]-R[i]-R[j];
// 				if (neighborDist_current < neighborDist){
// 					neighborDist = neighborDist_current;
//                     i_selected = i;
//                     j_selected = j;
// 				}
// 			}
// 		}
// 		vertexIndsForIterating.erase(remove(vertexIndsForIterating.begin(),vertexIndsForIterating.end(),i_selected),vertexIndsForIterating.end());
// 		vertexIndsForIterating.erase(remove(vertexIndsForIterating.begin(),vertexIndsForIterating.end(),j_selected),vertexIndsForIterating.end());		
// 		vertexIndsForIterating.push_back(h);
// 		SEM_vertex* h_ptr = new SEM_vertex (h,emptySequence);
// 		h_ptr->name = "h_" + to_string(this->h_ind);
// //		h_ptr->name = "h_" + to_string(h);		
// 		this->h_ind += 1;
// 		this->vertexMap->insert(pair<int,SEM_vertex*>(h,h_ptr));
// 		(*this->vertexMap)[i_selected]->AddNeighbor((*this->vertexMap)[h]);
// 		(*this->vertexMap)[j_selected]->AddNeighbor((*this->vertexMap)[h]);
// 		(*this->vertexMap)[h]->AddNeighbor((*this->vertexMap)[i_selected]);
// 		(*this->vertexMap)[h]->AddNeighbor((*this->vertexMap)[j_selected]);
// //		vertices.push_back(vertex_h_ptr);
// //		vertices[i_selected]->AddNeighbor(vertices[h]);
// //		vertices[j_selected]->AddNeighbor(vertices[h]);
// //		vertices[h]->AddNeighbor(vertices[i_selected]);
// //		vertices[h]->AddNeighbor(vertices[j_selected]);
//         R.erase(i_selected);
//         R.erase(j_selected);
//         R[h] = 0.0;
// 		n -= 1;
// 		for (unsigned int k_ind = 0; k_ind < n-1; k_ind++) {
// 			k = vertexIndsForIterating[k_ind];
// 			if (k < i_selected) {
// 				newDist = distanceMap[pair<int,int>(k,i_selected)] + distanceMap[pair<int,int>(k,j_selected)];
//                 newDist -= distanceMap[pair<int,int>(i_selected,j_selected)];
//                 newDist *= 0.5;
//                 R[k] = float(R[k]*(n-1)-distanceMap[pair<int,int>(k,i_selected)]-distanceMap[pair<int,int>(k,j_selected)] + newDist)/float(n-2);
//                 distanceMap.erase(pair<int,int>(k,i_selected));
// 				distanceMap.erase(pair<int,int>(k,j_selected));               
// 			} else if (j_selected < k) {
// 				newDist = distanceMap[pair<int,int>(i_selected,k)] + distanceMap[pair<int,int>(j_selected,k)];
//                 newDist -= distanceMap[pair<int,int>(i_selected,j_selected)];
//                 newDist *= 0.5;
//                 R[k] = float(R[k]*(n-1)-distanceMap[pair<int,int>(i_selected,k)]-distanceMap[pair<int,int>(j_selected,k)] + newDist)/float(n-2);
//                 distanceMap.erase(pair<int,int>(i_selected,k));
//                 distanceMap.erase(pair<int,int>(j_selected,k));
// 			} else {
// 			    newDist = distanceMap[pair<int,int>(i_selected,k)] + distanceMap[pair<int,int>(k,j_selected)];
//                 newDist -= distanceMap[pair<int,int>(i_selected,j_selected)];
//                 newDist *= 0.5;
//                 R[k] = float(R[k]*(n-1)-distanceMap[pair<int,int>(i_selected,k)]-distanceMap[pair<int,int>(k,j_selected)] + newDist)/float(n-2);
//                 distanceMap.erase(pair<int,int>(i_selected,k));
//                 distanceMap.erase(pair<int,int>(k,j_selected));
// 			}            
// 			distanceMap[pair<int,int>(k,h)] = newDist;
//             R[h] += newDist;
// 		}
//         R[h] /= float(n-2);
// 		h += 1;
//         distanceMap.erase(pair<int,int>(i_selected,j_selected));
// 	}
// 	SEM_vertex* h_ptr = new SEM_vertex (h,emptySequence);
// 	h_ptr->name = "h_" + to_string(this->h_ind);
// //	h_ptr->name = "h_" + to_string(h);
// 	this->h_ind += 1;
// 	this->vertexMap->insert(pair<int,SEM_vertex*>(h,h_ptr));
// 	for (int v:vertexIndsForIterating) {
// 		(*this->vertexMap)[v]->AddNeighbor((*this->vertexMap)[h]);
// 		(*this->vertexMap)[h]->AddNeighbor((*this->vertexMap)[v]);
// 	}
// }

#endif
