#include <pybind11/pybind11.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <math.h>  
#include <numeric> 
#include <random>
#include <sstream>
#include <stdio.h>
#include <string>
#include <tuple>
#include <vector>

using namespace std; 
namespace py = pybind11;

struct mt_error : runtime_error {
    using runtime_error::runtime_error;
};


int ll_precision = 24;

///...///...///...///...///...///... Dirichlet sampler ///...///...///...///...///...///...///...///

inline random_device rd;
inline mt19937_64 gen(rd());

inline array <double, 4> sample_dirichlet(const array<double, 4>& alpha, mt19937_64& gen) {
    array <double, 4> x;
    double sum = 0.0;
    for (size_t i = 0; i < 4; ++i) {
        gamma_distribution<double> gamma(alpha[i], 1.0);
        x[i] = gamma(gen);
        sum += x[i];
    }
    for (auto& v : x) v /= sum;
    return x;
}

array <double, 4> D_pi = {100, 100, 100, 100};
array <double, 4> D_M_row = {100, 2, 2, 2};

inline array <double, 4> sample_pi() {
	array <double, 4> pi = sample_dirichlet(D_pi,gen);
	return (pi);
}

inline array <double, 4> sample_M_row() {
	array <double, 4> pi = sample_dirichlet(D_M_row,gen);
	return (pi);
}


inline bool starts_with(const string& str, const string& prefix) {
    return str.size() >= prefix.size() &&
           str.compare(0, prefix.size(), prefix) == 0;
}

inline pair<int,int> ord_pair(int a, int b) {
    return (a < b) ? make_pair(a,b) : make_pair(b,a);
}

inline vector<string> split_ws(const string& s) {
    istringstream iss(s);
    vector <string> out;
    for (string tok; iss >> tok;) out.push_back(tok);
    return out;
}


typedef array<array<double,4>,4> Md;

inline Md MT(Md P) {
	Md P_t;
	for (int i = 0; i < 4; i++) {
		for (int j = 0; j < 4; j++) {
			P_t[j][i] = P[i][j];
		}
	}
	return (P_t);
}

inline Md MM(const Md &A, const Md &B) {
    Md result = {0.0}; // Initialize all elements to 0
    for (int i = 0; i < 4; i++) {
        for (int j = 0; j < 4; j++) {            
            for (int k = 0; k < 4; k++) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

///...///...///...///...///...///...///... struct for prim ...///...///...///...///...///...///...///
struct prim_graph {
    int n;
    vector<double> W;    
    
    prim_graph(int n_, const pair<int,int>* edges, const double* weights, int m) : n(n_), W(size_t(n_) * size_t(n_), numeric_limits<double>::infinity()) {        
        for (int i = 0; i < n; ++i) W[size_t(i)*n + i] = 0.0f;
        
        for (int k = 0; k < m; ++k) {
            int u = edges[k].first;
            int v = edges[k].second;
            double w = weights[k];
            if (!(0 <= u && u < n && 0 <= v && v < n)) {
                throw mt_error("check input for prim's algorithm");			
            }
            W[size_t(u)*n + v] = w;
            W[size_t(v)*n + u] = w;
        }
    }

    int num_vertices() const { return n; }
    
    double weight(int u, int v) const { return W[size_t(u)*n + v]; }
};

inline void prim(const prim_graph& g, int* parent_out) {
    const int n = g.num_vertices();
    
    const double INF = numeric_limits<double>::infinity();
    vector<double> key(n, INF);
    vector<char>  inMST(n, 0);

    // root = 0
    key[0] = 0.0f;
    parent_out[0] = 0;

    for (int it = 0; it < n; ++it) {
        int u = -1;
        double best = INF;
        for (int v = 0; v < n; ++v) {
            if (!inMST[v] && key[v] < best) {
                best = key[v];
                u = v;
            }
        }
        if (u == -1) break; 
        inMST[u] = 1;
        
        for (int v = 0; v < n; ++v) {
            if (inMST[v] || v == u) continue;
            double w = g.weight(u, v);
            if (w < key[v]) {
                key[v] = w;
                parent_out[v] = u;
            }
        }
    }
}

///...///...///...///...///...///...///... mst vertex ...///...///...///...///...///...///...///

class MST_vertex {
public:
	string name;
	int degree = 0;
	int numberOfLargeEdgesInSubtree = 0;
	int timesVisited = 0;
	int id;
	int idOfExternalVertex;
	int rank = 0;	
	vector <unsigned char> sequence;
	vector <unsigned char> globallyCompressedSequence;
	vector <int> idsOfVerticesInSubtree;
	vector <MST_vertex *> neighbors;
	vector <string> dupl_seq_names;
	void AddNeighbor(MST_vertex * v_ptr);		
	MST_vertex(int idToAdd, string nameToAdd, vector <unsigned char> sequenceToAdd) {
		id = idToAdd;
		sequence = sequenceToAdd;
		name = nameToAdd;
		idsOfVerticesInSubtree.push_back(idToAdd);
		idOfExternalVertex = -1;
	}
	~MST_vertex() {
		
	}
};

void MST_vertex::AddNeighbor(MST_vertex* v_ptr) {
	degree += 1;
	neighbors.push_back(v_ptr);
}

///...///...///...///...///...///...///...///... mst ...///...///...///...///...///...///...///...///

class MST {
private:	
	int largestVertexIndex;
	int edgeWeightThreshold = 0;
	vector <MST_vertex*> verticesToVisit;	
	bool ContainsVertex(int vertex_id);
	map <pair<int,int>,int> * allEdgeWeights;
	chrono::system_clock::time_point current_time;
	chrono::system_clock::time_point start_time;
	chrono::system_clock::time_point time_to_compute_MST;	
		
public:
	int maxDegree;	
	unsigned int num_duplicated_sequences = 0;
    int sequence_length;
	vector <int> siteWeights;
	string sequenceFileName;
	int v_ind;
	int numberOfLargeEdgesThreshold_input = 0;
	int numberOfLargeEdgesThreshold = 0;
	int numberOfNonZeroWeightEdges = 0;
    int numberOfInputSequences = 0;
	vector <int> idsOfExternalVertices;
	vector <MST_vertex *> leader_ptrs;  
	void SetLeaders();
	map <string, unsigned char> mapDNAtoInteger;	
	map <string, vector <string>> unique_seq_id_2_dupl_seq_ids;
	map <vector <unsigned char>, MST_vertex *> unique_seq_2_MST_vertex_ptr;
	MST_vertex * subtree_v_ptr;
	map <int, MST_vertex *> * vertexMap;
	map <pair <int, int>, int> edgeWeightsMap;
	string EncodeAsDNA(vector<unsigned char> sequence);
	vector<unsigned char> DecompressSequence(vector<unsigned char>* compressedSequence, vector<vector<int>>* sitePatternRepeats);
	void AddEdgeWeight(int u_id, int v_id, int edgeWeight);
	void RemoveEdgeWeight(int u_id, int v_id);
	void AddEdgeWeightToDistanceGraph(int u_id, int v_id, int edgeWeight);
	void RemoveWeightedEdgesIncidentToVertexInDistanceGraph(int u_id);
	void SetCompressedSequencesAndSiteWeightsForInputSequences();
	bool IsSequenceDuplicated(vector<unsigned char> sequence);
	void AddDuplicatedSequenceName(string name, vector<unsigned char> sequence);
	void SetNumberOfLargeEdgesThreshold(int numberOfLargeEdges);
	void SetEdgeWeightThreshold(int edgeWeight){edgeWeightThreshold = edgeWeight;}
	void AddVertex(string name, vector <unsigned char> sequence);
	void AddVertexWithId(int id, string name, vector <unsigned char> sequence);
	void RemoveVertex(int vertex_id);
	void AddEdge(int u_id, int v_id, int edgeWeight);
	void RemoveEdge(int u_id, int v_id);
	void ResetVertexAttributesForSubtreeSearch();
	void UpdateMSTWithMultipleExternalVertices(vector <int> idsOfVerticesToKeep, vector <int> idsOfVerticesToRemove, vector <tuple<int,string,vector<unsigned char>>> idAndNameAndSeqTupleForVerticesToAdd, vector <int> idsOfExternalVertices);
	void UpdateMaxDegree();
	void UpdateMSTWithOneExternalVertex(vector <int> idsOfVerticesToRemove, string nameOfSequenceToAdd, vector <unsigned char> sequenceToAdd);
	bool ContainsEdge(int u_id, int v_id);
	int GetEdgeWeight(int u_id, int v_id);
	int GetEdgeIndex (int vertexIndex1, int vertexIndex2, int numberOfVertices);
	int GetNumberOfVertices();
	void ReadFasta(string sequenceFileNameToSet);
    void ReadPhyx(string sequenceFileNameToSet);
	void ComputeMST();
	void CLGrouping();		
	void ResetSubtreeSizeThreshold();	
	void doubleSubtreeSizeThreshold();
	int ComputeHammingDistance(vector <unsigned char> recodedSeq1, vector <unsigned char> recodedSeq2);
	pair <vector <int>, vector <int>> GetIdsForSubtreeVerticesAndExternalVertices();
	pair <bool, MST_vertex *> GetPtrToVertexSubtendingSubtree();
	pair <vector <int>,vector <int>> GetSubtreeVerticesAndExternalVertices();
	tuple <vector <string>, vector <vector <unsigned char>>, vector <int>, vector <vector<int>>> GetCompressedSequencesSiteWeightsAndSiteRepeats(vector <int> vertexIdList);
	vector <int> GetIdsOfClosestUnvisitedVertices(MST_vertex* u_ptr);
	void SetIdsOfExternalVertices();
	bool ShouldIComputeALocalPhylogeneticTree();
	void WriteToFile(string fileName);
	unsigned char ConvertDNAToChar(char dna);	
	MST() {		
		this->v_ind = 0;		
		vector <unsigned char> emptySequence;
		this->allEdgeWeights = new map <pair<int,int>,int> ; 
		this->vertexMap = new map <int, MST_vertex *>;
		this->mapDNAtoInteger["A"] = 0;
		this->mapDNAtoInteger["C"] = 1;
		this->mapDNAtoInteger["G"] = 2;
		this->mapDNAtoInteger["T"] = 3;		
		this->mapDNAtoInteger["-"] = 4;
		this->mapDNAtoInteger["N"] = 4;
		this->mapDNAtoInteger["W"] = 4;
		this->mapDNAtoInteger["S"] = 4;
		this->mapDNAtoInteger["M"] = 4;
		this->mapDNAtoInteger["K"] = 4;
		this->mapDNAtoInteger["R"] = 4;
		this->mapDNAtoInteger["Y"] = 4;
		this->mapDNAtoInteger["B"] = 4;
		this->mapDNAtoInteger["D"] = 4;
		this->mapDNAtoInteger["H"] = 4;
		this->mapDNAtoInteger["V"] = 4;		
	
	}
	~MST() {		
		for (pair<int,MST_vertex*> VptrMap: *this->vertexMap){			
			delete VptrMap.second;
		}
		delete this->vertexMap;
		delete this->allEdgeWeights;
	}
};

void MST::SetLeaders() {
	leader_ptrs.clear();
	for (pair<int,MST_vertex*> VptrMap: *this->vertexMap) {					
		if (VptrMap.second->degree > 1) {
			leader_ptrs.push_back(VptrMap.second);
		}			
	}
}

bool MST::IsSequenceDuplicated(vector<unsigned char> query_seq) {
	if (this->unique_seq_2_MST_vertex_ptr.find(query_seq) != this->unique_seq_2_MST_vertex_ptr.end()) {
		return (true);
	} else {
		return (false);
	}
}

void MST::AddDuplicatedSequenceName(string dupl_seq_name, vector <unsigned char> sequence) {	
	MST_vertex * v = this->unique_seq_2_MST_vertex_ptr[sequence];
	this->unique_seq_id_2_dupl_seq_ids[v->name].push_back(dupl_seq_name);
	this->num_duplicated_sequences += 1;
}

void MST::AddEdgeWeightToDistanceGraph(int u_id, int v_id, int edgeWeight) {
	if (u_id < v_id) {
		this->allEdgeWeights->insert(make_pair(make_pair(u_id,v_id),edgeWeight));
	} else {
		this->allEdgeWeights->insert(make_pair(make_pair(v_id,u_id),edgeWeight));
	}
}

void MST::SetIdsOfExternalVertices() {
	this->idsOfExternalVertices.clear();
	this->idsOfExternalVertices = this->GetIdsOfClosestUnvisitedVertices(this->subtree_v_ptr);
}

int MST::GetEdgeIndex (int vertexIndex1, int vertexIndex2, int numberOfVertices) {
	int edgeIndex;
	edgeIndex = numberOfVertices * (numberOfVertices-1)/2;
	edgeIndex -= (numberOfVertices - vertexIndex1) * (numberOfVertices-vertexIndex1-1)/2;
	edgeIndex += vertexIndex2 - vertexIndex1 - 1;
	return edgeIndex;
}

void MST::SetNumberOfLargeEdgesThreshold(int numberOfLargeEdges_toSet) {
	this->numberOfLargeEdgesThreshold_input = numberOfLargeEdges_toSet;
	this->numberOfLargeEdgesThreshold = numberOfLargeEdges_toSet;
}

bool MST::ShouldIComputeALocalPhylogeneticTree() {
	bool valueToReturn;
	bool verbose = 1;
	bool subtreeExtractionPossible;
	int numberOfNonZeroWeightEdgesInVmWithoutVs;
	tie (subtreeExtractionPossible, this->subtree_v_ptr) = this->GetPtrToVertexSubtendingSubtree();	
	if (subtreeExtractionPossible) {
		numberOfNonZeroWeightEdgesInVmWithoutVs = this->numberOfNonZeroWeightEdges - this->subtree_v_ptr->numberOfLargeEdgesInSubtree;
		if (numberOfNonZeroWeightEdgesInVmWithoutVs > this->numberOfLargeEdgesThreshold) {
			valueToReturn = 1;			
		} else {
			// if (verbose) {
			// 	cout << "Case 1: subtree extraction possible but number of external vertices is too small" << endl;
			// }
			valueToReturn = 0;
			
		}
	} else {
		// if (verbose) {
		// 	cout << "Case 2: subtree extraction is not possible" << endl;
		// }
		valueToReturn = 0;
	}
	return (valueToReturn);
}



void MST::ResetSubtreeSizeThreshold() {
	this->numberOfLargeEdgesThreshold = this->numberOfLargeEdgesThreshold_input;
}

void MST::doubleSubtreeSizeThreshold() {
	this->numberOfLargeEdgesThreshold = this->numberOfLargeEdgesThreshold * 2;
}
int MST::ComputeHammingDistance(vector<unsigned char> recodedSeq1, vector<unsigned char> recodedSeq2) {	
	int hammingDistance = 0;	
	for (unsigned int i=0;i<recodedSeq1.size();i++){
		if (recodedSeq1[i] == 4 or recodedSeq1[i] == 4) {
			continue;
		} else {			
			if (recodedSeq1[i] != recodedSeq2[i]) {
					hammingDistance+=1.0;
				}
		}		
	}
	return (hammingDistance);
};


int MST::GetNumberOfVertices() {
	return this->vertexMap->size();
};


bool MST::ContainsVertex(int vertex_id) {
	return this->vertexMap->find(vertex_id)!=vertexMap->end();
}

void MST::AddVertex(string name, vector <unsigned char> sequence) {
	MST_vertex * v = new MST_vertex(this->v_ind, name, sequence);
	this->vertexMap->insert(pair<int,MST_vertex*>(this->v_ind,v));
	this->unique_seq_2_MST_vertex_ptr[sequence] = v;	
	this->v_ind += 1;
}

void MST::AddVertexWithId(int id, string name, vector <unsigned char> sequence) {
	MST_vertex * v = new MST_vertex(id, name, sequence);
	this->vertexMap->insert(pair<int,MST_vertex*>(id,v));
}

void MST::RemoveVertex(int vertex_id) {
	MST_vertex* v = (*this->vertexMap)[vertex_id];
	for (MST_vertex* n: v->neighbors) {
		if (n->id < v->id) {
			this->edgeWeightsMap.erase(pair<int,int>(n->id,v->id));
		} else {
			this->edgeWeightsMap.erase(pair<int,int>(v->id,n->id));
		}
		n->neighbors.erase(remove(n->neighbors.begin(),n->neighbors.end(),v),n->neighbors.end());
		n->degree -= 1;
	}
	v->neighbors.clear();
	v->sequence.clear();
	v->idsOfVerticesInSubtree.clear();
	this->vertexMap->erase(vertex_id);
	delete v;
}

void MST::AddEdge(int u_id, int v_id, int edgeWeight) {
	MST_vertex* u_ptr = (*this->vertexMap)[u_id];
	MST_vertex* v_ptr = (*this->vertexMap)[v_id];
	u_ptr->AddNeighbor(v_ptr);
	v_ptr->AddNeighbor(u_ptr);
	this->AddEdgeWeight(u_id,v_id,edgeWeight);
	if (edgeWeight > 0) {
		this->numberOfNonZeroWeightEdges += 1;
	}
};

void MST::RemoveEdge(int u_id, int v_id) {	
	if (u_id < v_id) {
		this->edgeWeightsMap.erase(pair<int,int>(u_id, v_id));
	} else {
		this->edgeWeightsMap.erase(pair<int,int>(v_id, u_id));
	}
	MST_vertex * u = (*this->vertexMap)[u_id];
	MST_vertex * v = (*this->vertexMap)[v_id];
	u->neighbors.erase(remove(u->neighbors.begin(),u->neighbors.end(),v),u->neighbors.end());
	u->degree -= 1;
	v->neighbors.erase(remove(v->neighbors.begin(),v->neighbors.end(),u),v->neighbors.end());
	v->degree -= 1;
}

bool MST::ContainsEdge(int u_id, int v_id) {
	if (u_id < v_id) {
		return (this->edgeWeightsMap.find(pair<int,int>(u_id,v_id)) != this->edgeWeightsMap.end());
	} else {
		return (this->edgeWeightsMap.find(pair<int,int>(v_id,u_id)) != this->edgeWeightsMap.end());
	}	
}

int MST::GetEdgeWeight(int u_id, int v_id) {
	if (u_id < v_id) {
		return this->edgeWeightsMap[pair<int,int>(u_id,v_id)];
	} else {
		return this->edgeWeightsMap[pair<int,int>(v_id,u_id)];
	}
}

void MST::AddEdgeWeight(int u_id, int v_id, int edgeWeight) {
	pair<int,int> edge ;
	if (u_id < v_id){
		edge = make_pair(u_id,v_id);
	} else {
		edge = make_pair(v_id,u_id);
	}
	if (this->edgeWeightsMap.find(edge) != this->edgeWeightsMap.end()) {
		this->edgeWeightsMap[edge] = edgeWeight;
	} else {
		this->edgeWeightsMap.insert(make_pair(edge,edgeWeight));
	}	
}

void MST::RemoveEdgeWeight(int u_id, int v_id) {
	pair <int, int> edge;
	if (u_id < v_id){
		edge = make_pair(u_id,v_id);
	} else {
		edge = make_pair(v_id,u_id);
	}
	this->edgeWeightsMap.erase(edge);
}

vector<int> MST::GetIdsOfClosestUnvisitedVertices(MST_vertex* v_ptr) {
	int numberOfLargeEdgesEncountered = 0;
	vector <int> idsOfClosestUnvisitedVertices;
	vector <MST_vertex*> verticesInCurrentLevel;
	for (MST_vertex* n_ptr: v_ptr->neighbors) {		
		if (find(v_ptr->idsOfVerticesInSubtree.begin(),v_ptr->idsOfVerticesInSubtree.end(),n_ptr->id)==v_ptr->idsOfVerticesInSubtree.end()) {
			idsOfClosestUnvisitedVertices.push_back(n_ptr->id);
			if (this->GetEdgeWeight(v_ptr->id,n_ptr->id)  > edgeWeightThreshold) {
				numberOfLargeEdgesEncountered+=1;
			}
			if (numberOfLargeEdgesEncountered < numberOfLargeEdgesThreshold) {
				verticesInCurrentLevel.push_back(n_ptr);
			}
		}
	}
	vector <MST_vertex *> verticesInNextLevel;
	while (numberOfLargeEdgesEncountered < numberOfLargeEdgesThreshold and verticesInCurrentLevel.size() > 0) {
		for(MST_vertex * x_ptr:verticesInCurrentLevel) {
			for (MST_vertex * n_ptr : x_ptr->neighbors) {
				if (find(idsOfClosestUnvisitedVertices.begin(),idsOfClosestUnvisitedVertices.end(),n_ptr->id)==idsOfClosestUnvisitedVertices.end() and n_ptr->id!=v_ptr->id) {
					idsOfClosestUnvisitedVertices.push_back(n_ptr->id);
					if(this->GetEdgeWeight(x_ptr->id,n_ptr->id) > edgeWeightThreshold) {
						numberOfLargeEdgesEncountered+=1;
					}
					if (numberOfLargeEdgesEncountered < numberOfLargeEdgesThreshold) {
						verticesInNextLevel.push_back(n_ptr);	
					}
				}
			}
		}
		verticesInCurrentLevel = verticesInNextLevel;
		verticesInNextLevel.clear();
	}
	return idsOfClosestUnvisitedVertices;
}

pair <bool, MST_vertex *> MST::GetPtrToVertexSubtendingSubtree() {
	this->ResetVertexAttributesForSubtreeSearch();	
	bool subTreeFound = 0;
	verticesToVisit.clear();
	for (pair <int, MST_vertex *> VptrMap: * this->vertexMap) {
		if (VptrMap.second->degree == 1) {
			verticesToVisit.push_back(VptrMap.second);
		}
	}
	vector <MST_vertex *> verticesVisited;
	int vertex_ind = verticesToVisit.size() -1;	
	while(vertex_ind > -1 and !subTreeFound) {
		this->subtree_v_ptr = verticesToVisit[vertex_ind];
		verticesToVisit.pop_back();
		vertex_ind -= 1;
		this->subtree_v_ptr->timesVisited += 1;
		for (MST_vertex * neighbor_ptr : this->subtree_v_ptr->neighbors) {
			if (neighbor_ptr->timesVisited < neighbor_ptr->degree) {
				neighbor_ptr->timesVisited += 1;
				for (int n_id : this->subtree_v_ptr->idsOfVerticesInSubtree) {
					neighbor_ptr->idsOfVerticesInSubtree.push_back(n_id);
				}
				neighbor_ptr->numberOfLargeEdgesInSubtree += this->subtree_v_ptr->numberOfLargeEdgesInSubtree;
				if (GetEdgeWeight(this->subtree_v_ptr->id,neighbor_ptr->id) > edgeWeightThreshold) {
					neighbor_ptr->numberOfLargeEdgesInSubtree+=1;
				}
				if (neighbor_ptr->degree - neighbor_ptr->timesVisited == 1) {
					if (neighbor_ptr->numberOfLargeEdgesInSubtree > numberOfLargeEdgesThreshold) {
						subTreeFound = 1;
						// set id to external vertex
						for (MST_vertex * v : neighbor_ptr->neighbors) {
							if (v->timesVisited < v->degree) {
								neighbor_ptr->idOfExternalVertex = v->id;
							}							
						}						
						return pair <bool, MST_vertex *> (subTreeFound,neighbor_ptr);						
					}
					verticesToVisit.push_back(neighbor_ptr);
					vertex_ind+=1;
				}
			}
		}
	}	
	return pair <bool, MST_vertex *> (subTreeFound,this->subtree_v_ptr);
}

void MST::ResetVertexAttributesForSubtreeSearch() {
	for (pair <int, MST_vertex *> VIdAndPtr: *this->vertexMap) {
		VIdAndPtr.second->numberOfLargeEdgesInSubtree = 0;
		VIdAndPtr.second->idsOfVerticesInSubtree.clear();
		VIdAndPtr.second->idsOfVerticesInSubtree.push_back(VIdAndPtr.second->id);
		VIdAndPtr.second->timesVisited = 0;
	}
}


void MST::UpdateMSTWithOneExternalVertex(vector<int> idsOfVerticesToRemove, string nameOfSequenceToAdd, vector <unsigned char> sequenceToAdd) {
	//	Remove vertices		
	for (int v_id: idsOfVerticesToRemove) {	
		this->RemoveVertex(v_id);
	}
	//	Remove neighbors and reset vertex attributes
	for (pair<int,MST_vertex*> VIdAndPtr: *this->vertexMap) {
		VIdAndPtr.second->numberOfLargeEdgesInSubtree = 0;
		VIdAndPtr.second->idsOfVerticesInSubtree.clear();
		VIdAndPtr.second->idsOfVerticesInSubtree.push_back(VIdAndPtr.second->id);
		VIdAndPtr.second->timesVisited = 0;
		VIdAndPtr.second->neighbors.clear();
		VIdAndPtr.second->degree = 0;
	}
	this->numberOfNonZeroWeightEdges = 0;
	// Add vertex
	int indexOfVertexToAdd = this->v_ind;
	this->AddVertex(nameOfSequenceToAdd, sequenceToAdd);
	MST_vertex * v_add; MST_vertex * v_inMST;
	v_add = (*this->vertexMap)[indexOfVertexToAdd];
	int edgeWeight;
	for (pair <int, MST_vertex *> idPtrPair : *this->vertexMap) {
		if (idPtrPair.first != indexOfVertexToAdd) {
			v_inMST = idPtrPair.second;
			edgeWeight = ComputeHammingDistance(v_add->sequence, v_inMST->sequence);
			if (v_inMST->id < v_add->id){					
				this->edgeWeightsMap[pair<int,int>(v_inMST->id,v_add->id)] = edgeWeight;					
			} else {					
				this->edgeWeightsMap[pair<int,int>(v_add->id,v_inMST->id)] = edgeWeight;					
			}
		}
	}	
	int numberOfVertices = int(this->vertexMap->size());	
	const int numberOfEdges = int(this->edgeWeightsMap.size());
	
	double * weights;
	weights = new double [numberOfEdges];
	
	typedef pair <int,int > E;
	E * edges;
	edges = new E [numberOfEdges];
	
	int edgeIndex = 0;
    for (pair<pair<int,int>,int> edgeAndWeight : this->edgeWeightsMap) {
		edges[edgeIndex] = E(edgeAndWeight.first.first,edgeAndWeight.first.second);
		weights[edgeIndex] = edgeAndWeight.second;
		edgeIndex += 1;		
	}
	
	vector<int> p(numberOfVertices); 

	prim_graph p_graph(numberOfVertices, edges, weights, numberOfEdges);
	
	prim(p_graph, &p[0]);
	
	delete[] edges;		
	delete[] weights;		
	vector <pair<int,int>> edgeWeightsToKeep;	
	vector <pair<int,int>> edgeWeightsToRemove;	
	for (size_t u = 0; u != p.size(); ++u) {
		if (p[u] != u) {		
			this->AddEdge(u,p[u],this->GetEdgeWeight(u,p[u]));
			if (u < p[u]) {
				edgeWeightsToKeep.push_back(pair<int,int>(u,p[u]));
			} else {
				edgeWeightsToKeep.push_back(pair<int,int>(p[u],u));
			}
		}
	}
	for (pair<pair<int,int>,int> edgeAndWeight : this->edgeWeightsMap) {
		if (find(edgeWeightsToKeep.begin(),edgeWeightsToKeep.end(),edgeAndWeight.first) == edgeWeightsToKeep.end()){
			edgeWeightsToRemove.push_back(edgeAndWeight.first);
		}
	}
	for (pair<int,int> edge: edgeWeightsToRemove) {
		this->edgeWeightsMap.erase(edge);
	}	
}

void MST::UpdateMaxDegree() {
	this->maxDegree = 0;
	for (pair <int,MST_vertex*> VIdAndPtr: *this->vertexMap) {
		if (this->maxDegree	< VIdAndPtr.second->degree) {
			this->maxDegree	= VIdAndPtr.second->degree;
		}
	}
}

void MST::UpdateMSTWithMultipleExternalVertices(vector<int> idsOfVerticesToKeep, vector<int> idsOfVerticesToRemove, vector<tuple<int,string,vector<unsigned char>>> idAndNameAndSeqTupleForVerticesToAdd, vector<int> idsOfExternalVertices) {
	// Remove weights of edges incident to vertex
	//	Remove vertices		
	for (int v_id: idsOfVerticesToRemove) {
		this->RemoveVertex(v_id);
	}
	//	Remove all edges in MST and reset all attributes for each vertex
	for (pair <int,MST_vertex*> VIdAndPtr: *this->vertexMap) {
		VIdAndPtr.second->numberOfLargeEdgesInSubtree = 0;
		VIdAndPtr.second->idsOfVerticesInSubtree.clear();
		VIdAndPtr.second->idsOfVerticesInSubtree.push_back(VIdAndPtr.second->id);
		VIdAndPtr.second->timesVisited = 0;
		VIdAndPtr.second->neighbors.clear();
		VIdAndPtr.second->degree = 0;
	}
	this->numberOfNonZeroWeightEdges = 0;
	int u_id; int v_id; int edgeWeight;
	vector <unsigned char> seq_u; vector <unsigned char> seq_v;
	string u_name; string v_name;
	
	int numberOfVerticesToKeep = int(idsOfVerticesToKeep.size());
	int numberOfVerticesToAdd = int(idAndNameAndSeqTupleForVerticesToAdd.size());
	int numberOfExternalVertices = int(idsOfExternalVertices.size());
	
	if (numberOfVerticesToAdd > 1) {
		for (int u_ind = 0; u_ind < numberOfVerticesToAdd -1; u_ind++) {
			tie (u_id, u_name, seq_u) = idAndNameAndSeqTupleForVerticesToAdd[u_ind];
			for (int v_ind = u_ind + 1; v_ind < numberOfVerticesToAdd; v_ind++) {
				tie (v_id, v_name, seq_v) = idAndNameAndSeqTupleForVerticesToAdd[v_ind];
				edgeWeight = ComputeHammingDistance(seq_u, seq_v);
				this->AddEdgeWeight(u_id,v_id,edgeWeight);
			}
		}
	}
// Add newly introduced vertices
	for (int u_ind = 0; u_ind < numberOfVerticesToAdd; u_ind++) {
		tie (u_id, u_name, seq_u) = idAndNameAndSeqTupleForVerticesToAdd[u_ind];
		this->AddVertexWithId(u_id, u_name, seq_u);
	}
// Add edge weights for vertices to add to external vertices
	for (int u_ind = 0; u_ind < numberOfVerticesToAdd; u_ind++) {
		tie (u_id, u_name, seq_u) = idAndNameAndSeqTupleForVerticesToAdd[u_ind];
		for (int v_ind = 0; v_ind < numberOfExternalVertices; v_ind++) {
			v_id = idsOfExternalVertices[v_ind];
			seq_v = (*this->vertexMap)[v_id]->sequence;
			edgeWeight = ComputeHammingDistance(seq_u,seq_v);
			this->AddEdgeWeight(u_id,v_id,edgeWeight);
		}
	}
		
	for (int u_ind = 0; u_ind < numberOfVerticesToAdd; u_ind++) {
		tie (u_id, u_name, seq_u) = idAndNameAndSeqTupleForVerticesToAdd[u_ind];		
		for (int v_ind = 0; v_ind < numberOfVerticesToKeep; v_ind++) {
			v_id = idsOfVerticesToKeep[v_ind];
			seq_v = (*this->vertexMap)[v_id]->sequence;
			edgeWeight = ComputeHammingDistance(seq_u,seq_v);
			this->AddEdgeWeight(u_id,v_id,edgeWeight);
		}		
	}

	for (int u_ind = 0; u_ind < numberOfVerticesToKeep; u_ind++) {
			u_id = idsOfVerticesToKeep[u_ind];
			seq_u = (*this->vertexMap)[u_id]->sequence;
		for (int v_ind = 0; v_ind < numberOfExternalVertices; v_ind++) {
			v_id = idsOfExternalVertices[v_ind];
			seq_v = (*this->vertexMap)[v_id]->sequence;
			edgeWeight = ComputeHammingDistance(seq_u,seq_v);
			this->AddEdgeWeight(u_id,v_id,edgeWeight);
		}
	}
		
	vector <int> mstIds;
	map<int,int> mstId2PrimId;
	int primId = 0;
	int mstId;
	for (pair <int,MST_vertex*> idPtrPair : *this->vertexMap) {
		mstId = idPtrPair.first;
		mstIds.push_back(mstId);
		mstId2PrimId.insert(make_pair(mstId,primId));
		primId += 1;
	}
	int numberOfVertices = int(this->vertexMap->size());	

	const int numberOfEdges = int(this->edgeWeightsMap.size());
	double * weights;
	weights = new double [numberOfEdges];
	
	typedef pair <int,int > E;
	E * edges;
	edges = new E [numberOfEdges];
		
	
	int edgeIndex = 0;	
    for (pair<pair<int,int>,int> edgeAndWeight : this->edgeWeightsMap) {
		tie (u_id, v_id) = edgeAndWeight.first;
		edges[edgeIndex] = E(mstId2PrimId[u_id],mstId2PrimId[v_id]);
		weights[edgeIndex] = edgeAndWeight.second;
		edgeIndex += 1;		
	}
	
	vector<int> p(numberOfVertices); 

	prim_graph p_graph(numberOfVertices, edges, weights, numberOfEdges);
	
	prim(p_graph, &p[0]);
				
	vector <pair<int,int>> edgeWeightsToKeep;	
	vector <pair<int,int>> edgeWeightsToRemove;	

	for (size_t u = 0; u != p.size(); ++u) {
		if (p[u] != u){
			u_id = mstIds[p[u]];
			v_id = mstIds[u];
			this->AddEdge(u_id,v_id,this->GetEdgeWeight(u_id,v_id));
			if (u_id < v_id){
				edgeWeightsToKeep.push_back(pair<int,int>(u_id,v_id));
			} else {
				edgeWeightsToKeep.push_back(pair<int,int>(v_id,u_id));
			}
		}
	}
	this->UpdateMaxDegree();
	
	if (this->maxDegree == 0) {
		ofstream edgeListFile;
		edgeListFile.open(sequenceFileName+".debugEdgeList");
		for (pair<pair<int,int>,int> edgeAndWeight : this->edgeWeightsMap) {
			edgeListFile << edgeAndWeight.first.first << "\t";
			edgeListFile << edgeAndWeight.first.second << "\t";
			edgeListFile << edgeAndWeight.second << endl;
		}
		edgeListFile.close();
	}
	
	delete[] edges;
	delete[] weights;
	for (pair<pair<int,int>,int> edgeAndWeight : this->edgeWeightsMap) {
		if (find(edgeWeightsToKeep.begin(),edgeWeightsToKeep.end(),edgeAndWeight.first) == edgeWeightsToKeep.end()){
			edgeWeightsToRemove.push_back(edgeAndWeight.first);
		}
	}
	for (pair<int,int> edge: edgeWeightsToRemove) {
		this->edgeWeightsMap.erase(edge);
	}
}

tuple <vector<string>,vector<vector<unsigned char>>,vector<int>,vector<vector<int>>> MST::GetCompressedSequencesSiteWeightsAndSiteRepeats(vector<int> vertexIdList){	
	vector <string> names;
	vector <vector<unsigned char>> compressedSequences;
	vector <int> sitePatternWeights_ptr;
	vector <vector <int>> sitePatternRepeats_ptr;
	vector <vector<unsigned char>> distinctPatterns;
	map <vector<unsigned char>,vector<int>> distinctPatternsToSitesWherePatternRepeats;
	vector <MST_vertex*> vertexPtrList;
	for (unsigned int i = 0; i < vertexIdList.size(); i++){		
		MST_vertex* v_ptr = (*this->vertexMap)[vertexIdList[i]];
		vertexPtrList.push_back(v_ptr);
		vector <unsigned char> compressedSequence;
		compressedSequences.push_back(compressedSequence);
		names.push_back(v_ptr->name);
	}
	int numberOfSites = vertexPtrList[0]->sequence.size();
	vector<unsigned char> sitePattern;
	for(int site = 0; site < numberOfSites; site++){
		sitePattern.clear();
		for (MST_vertex* v_ptr: vertexPtrList){
			sitePattern.push_back(v_ptr->sequence[site]);}
		if (find(distinctPatterns.begin(),distinctPatterns.end(),sitePattern)!=distinctPatterns.end()){
			distinctPatternsToSitesWherePatternRepeats[sitePattern].push_back(site);
			
		} else {
			distinctPatterns.push_back(sitePattern);	
			vector<int> sitePatternRepeats;
			sitePatternRepeats.push_back(site);
			distinctPatternsToSitesWherePatternRepeats[sitePattern] = sitePatternRepeats;						
			for (unsigned int i = 0; i < sitePattern.size(); i++){
				compressedSequences[i].push_back(sitePattern[i]);
			}
		}
	}
	for (vector<unsigned char> sitePattern : distinctPatterns){
		int sitePatternWeight = distinctPatternsToSitesWherePatternRepeats[sitePattern].size();
		sitePatternWeights_ptr.push_back(sitePatternWeight);		
		sitePatternRepeats_ptr.push_back(distinctPatternsToSitesWherePatternRepeats[sitePattern]);
	}
	return make_tuple(names,compressedSequences,sitePatternWeights_ptr,sitePatternRepeats_ptr);
}

void MST::SetCompressedSequencesAndSiteWeightsForInputSequences() {				
	vector <vector<unsigned char>> distinctPatterns;
	map <vector<unsigned char>,vector<int>> distinctPatternsToSitesWherePatternRepeats;	
	int numberOfSites = (*this->vertexMap)[0]->sequence.size();
	int numberOfInputSequences = this->vertexMap->size();
	int sitePatternWeight; int v_id; int site;
	vector<unsigned char> sitePattern;
	for(site=0; site < numberOfSites; site++) {
		sitePattern.clear();
		for (v_id = 0; v_id < numberOfInputSequences; v_id ++) {
			sitePattern.push_back((*this->vertexMap)[v_id]->sequence[site]);
			}
		if (find(distinctPatterns.begin(),distinctPatterns.end(),sitePattern)!=distinctPatterns.end()) {
			distinctPatternsToSitesWherePatternRepeats[sitePattern].push_back(site);			
		} else {
			distinctPatterns.push_back(sitePattern);	
			vector<int> sitePatternRepeats;
			sitePatternRepeats.push_back(site);
			distinctPatternsToSitesWherePatternRepeats[sitePattern] = sitePatternRepeats;						
			for (v_id = 0; v_id < numberOfInputSequences; v_id ++) {				
				(*this->vertexMap)[v_id]->globallyCompressedSequence.push_back(sitePattern[v_id]);
			}
		}
	}
	for (vector<unsigned char> sitePattern: distinctPatterns) {
		sitePatternWeight = distinctPatternsToSitesWherePatternRepeats[sitePattern].size();		
		this->siteWeights.push_back(sitePatternWeight);		
	}
}

void MST::WriteToFile(string FileName) {
	ofstream mstFile;	
	mstFile.open(FileName);	
	MST_vertex * v;
	for (pair <int, MST_vertex *> vIdAndPtr: *this->vertexMap) {
		v = vIdAndPtr.second;
		for (MST_vertex * n: v->neighbors) {
			if (v->id < n->id) {
				mstFile << v->name << "\t" << n->name << "\t" << this->GetEdgeWeight(v->id, n->id) << endl; 
			}
		}
	}
	mstFile.close();
}

unsigned char MST::ConvertDNAToChar(char dna) {
	string dna_upper = string(1,toupper(dna));
	unsigned char dna_char = 4;
	if (this->mapDNAtoInteger.find(dna_upper) != this->mapDNAtoInteger.end()) {
		dna_char = this->mapDNAtoInteger[dna_upper];
	} else {
		if (isspace(dna)) {
			cout << "DNA character is a whitespace" << endl;
		}
		cout << "DNA character " << dna_upper << " is not in dictionary keys" << endl;
	}	
	return (dna_char);
}

void MST::ReadPhyx(string sequenceFileNameToSet) {
    this->sequenceFileName = sequenceFileNameToSet;
    ifstream inputFile(this->sequenceFileName.c_str());    
    vector <unsigned char> recodedSequence;
    string seqName;
    string seq = "";
    string line; getline(inputFile, line);
    vector <string> splitLine = split_ws(line);
    this->numberOfInputSequences = stoi(splitLine[0]);
    this->sequence_length = stoi(splitLine[1]);
    cout << "phylip file contains " << this->numberOfInputSequences << " sequences of length " << this->sequence_length << endl;
    for(line; getline(inputFile,line);) {
        vector <string> splitLine = split_ws(line);
        seqName = splitLine[0];
        seq = splitLine[1];
        for (char const dna: seq) {
            recodedSequence.push_back(this->ConvertDNAToChar(dna));
        }
        this->AddVertex(seqName,recodedSequence);
        recodedSequence.clear();
    }    
	inputFile.close();
    this->numberOfInputSequences = this->vertexMap->size();
}

void MST::ReadFasta(string sequenceFileNameToSet) {
	this->sequenceFileName = sequenceFileNameToSet;
	vector <unsigned char> recodedSequence;
	recodedSequence.clear();
	unsigned int site = 0;
    unsigned int seq_len = 0;
	unsigned char dna_char;
	int num_amb = 0;
	int num_non_amb = 0;
	ifstream inputFile(this->sequenceFileName.c_str());
	string seqName;
	string seq = "";
	for(string line; getline(inputFile, line );) {
		if (line[0]=='>') {
			if (seq != "") {
				for (char const dna: seq) {
					if (!isspace(dna)) {
						dna_char = this->ConvertDNAToChar(dna);
						if (dna_char > 3) {
							num_amb += 1;
						} else {
							num_non_amb += 1;
						}
						recodedSequence.push_back(dna_char);					
						site += 1;							
						}						
				}
				if (this->IsSequenceDuplicated(recodedSequence)) {
					this->AddDuplicatedSequenceName(seqName,recodedSequence);
				} else {										
					this->AddVertex(seqName,recodedSequence);
				}
				recodedSequence.clear();
			} 
			seqName = line.substr(1,line.length());
			seq = "";
			site = 0;			
		}
		else {
			seq += line ;
		}		
	}		
	for (char const dna: seq) {
		if (!isspace(dna)) {
			dna_char = this->ConvertDNAToChar(dna);
			if (dna_char > 3) { // FIX_AMB
				num_amb += 1;
			} else {
				num_non_amb += 1;
			}
			recodedSequence.push_back(dna_char);
			site += 1;
		}
	}
	if (this->IsSequenceDuplicated(recodedSequence)) {
		this->AddDuplicatedSequenceName(seqName,recodedSequence);
	} else {
		this->AddVertex(seqName,recodedSequence);
	}
    seq_len = recodedSequence.size();
	recodedSequence.clear();
	inputFile.close();
    cout << "Number of sequences in fasta file is " << this->vertexMap->size() << endl;
    cout << "Sequence length is " << seq_len << endl;
	// cout << "Number of ambiguous characters is " << double(num_amb) << "\tNumber of nonambiguous characters is " << double(num_non_amb) << endl;
	// cout << "Fraction of ambiguous characters is " << double(num_amb)/double(num_amb + num_non_amb) << endl;
}

vector<unsigned char> MST::DecompressSequence(vector<unsigned char>* compressedSequence, vector<vector<int>>* sitePatternRepeats){
	int totalSequenceLength = 0;
	for (vector<int> sitePatternRepeat: *sitePatternRepeats){
		totalSequenceLength += int(sitePatternRepeat.size());
	}
	vector <unsigned char> decompressedSequence;
	for (int v_ind = 0; v_ind < totalSequenceLength; v_ind++){
		decompressedSequence.push_back(char(0));
	}
	unsigned char dnaToAdd;
	for (int sitePatternIndex = 0; sitePatternIndex < int(compressedSequence->size()); sitePatternIndex++){
		dnaToAdd = (*compressedSequence)[sitePatternIndex];
		for (int pos: (*sitePatternRepeats)[sitePatternIndex]){
			decompressedSequence[pos] = dnaToAdd;
		}
	}
	return (decompressedSequence);	
}

string MST::EncodeAsDNA(vector<unsigned char> sequence){
	string allDNA = "AGTC";
	string dnaSequence = "";
	for (unsigned char s : sequence){
		dnaSequence += allDNA[s];
	}
	return dnaSequence;
}


void MST::ComputeMST() {

	int numberOfVertices = (this->v_ind);		
	const int numberOfEdges = numberOfVertices*(numberOfVertices-1)/2;		
	
	double * weights;
	weights = new double [numberOfEdges];
		
	int edgeIndex = 0;
	for (int i=0; i<numberOfVertices; i++) {
		for (int j=i+1; j<numberOfVertices; j++) {			
			weights[edgeIndex] = this->ComputeHammingDistance((*this->vertexMap)[i]->sequence,(*this->vertexMap)[j]->sequence);
			if (weights[edgeIndex] == 0) {
				cout << this->EncodeAsDNA((*this->vertexMap)[i]->sequence) << endl;
			}
			edgeIndex += 1;
		}
	}
	typedef pair <int,int > E;

	E * edges;
	edges = new E [numberOfEdges];
	edgeIndex = 0;
	for (int i=0; i<numberOfVertices; i++) {
		for (int j=i+1; j<numberOfVertices; j++) {
			edges[edgeIndex] = E(i,j);
			edgeIndex += 1;
		}
	}	
	
	vector<int> p(numberOfVertices); 

	prim_graph p_graph(numberOfVertices, edges, weights, numberOfEdges);
	
	prim(p_graph, &p[0]);
	delete[] edges;		
	int edgeCount = 0;
	for (size_t u = 0; u != p.size(); u++) {
		if (p[u] != u) {
			edgeCount += 1;
			if (u < p[u]) {
				edgeIndex = GetEdgeIndex(u,p[u],numberOfVertices);
			} else {
				edgeIndex = GetEdgeIndex(p[u],u,numberOfVertices);
			}
			this->AddEdge(u, p[u], weights[edgeIndex]);
		}
	}
	this->UpdateMaxDegree();
	delete[] weights;
}

///...///...///...///...///...///...///...///...///... structural expectation maximization ///...///...///...///...///...///...///...///...///

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
	double logScalingFactors = 0;
	double vertexLogLikelihood = 0;
	double sumOfEdgeLogLikelihoods = 0;
	int rateCategory = 0;
	int GCContent = 0;
	vector <SEM_vertex *> neighbors;
	vector <SEM_vertex *> children;
	array <double, 4> root_prob_hss;
	SEM_vertex * parent = this;
	void AddNeighbor(SEM_vertex * v_ptr);
	void RemoveNeighbor(SEM_vertex * v_ptr);
	void AddParent(SEM_vertex * v_ptr);
	void RemoveParent();
	void AddChild(SEM_vertex * v_ptr);
	void RemoveChild(SEM_vertex * v_ptr);
	void SetVertexLogLikelihood(double vertexLogLikelihoodToSet);
	int inDegree = 0;
	int outDegree = 0;
	Md transitionMatrix;
	Md transitionMatrix_stored;	
	array <double, 4> rootProbability;
	array <double, 4> posteriorProbability;	
	SEM_vertex (int idToAdd, vector <unsigned char> compressedSequenceToAdd) {
		this->id = idToAdd;
		this->compressedSequence = compressedSequenceToAdd;
		this->transitionMatrix = Md{};
		this->transitionMatrix_stored = Md{};
		for (int dna = 0; dna < 4; dna ++) {
			this->transitionMatrix[dna][dna] = 1.0;
			this->transitionMatrix_stored[dna][dna] = 1.0;
		}
		for (int i = 0; i < 4; i ++) {
			this->rootProbability[i] = 0;
			this->posteriorProbability[i] = 0;
			this->root_prob_hss[i] = 0;
		}
	}	
	~SEM_vertex () {
		this->neighbors.clear();
	}
};

void SEM_vertex::SetVertexLogLikelihood(double vertexLogLikelihoodToSet) {
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

///...///...///...///...///...///...///... clique ...///...///...///...///...///...///...///

class clique {
	public:	
	map <clique *, double> logScalingFactorForMessages;
	double logScalingFactorForClique;
	map <clique *, std::array <double, 4>> messagesFromNeighbors;
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
	std::array <double, 4> MarginalizeOverVariable(SEM_vertex * v);
//	Md DivideBeliefByMessageMarginalizedOverVariable(SEM_vertex * v);	
	// Clique is defined over the vertex pair (X,Y)
	// No of variables is always 2 for bifurcating tree-structured DAGs
	
	Md initialPotential;	
	Md belief;
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



std::array <double, 4> clique::MarginalizeOverVariable(SEM_vertex * v) {
	std::array <double, 4> message;	
	if (this->x == v) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			message[dna_y] = 0;
			for (int dna_x = 0; dna_x < 4; dna_x ++) {
				message[dna_y] += this->belief[dna_x][dna_y];
			}
		}
	} else if (this->y == v) {
		for (int dna_x = 0; dna_x < 4; dna_x ++) {
			message[dna_x] = 0;
			for (int dna_y = 0; dna_y < 4; dna_y ++) {
				message[dna_x] += this->belief[dna_x][dna_y];
			}
		}
	} else {
		cout << "Check marginalization over variable" << endl;
	}
	return (message);
}

void clique::ComputeBelief() {
	Md factor = this->initialPotential;
	vector <clique *> neighbors = this->children;
	std::array <double, 4> messageFromNeighbor;
	bool debug = 0;		
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
					factor[dna_x][dna_y] *= messageFromNeighbor[dna_y];					
				}
			}
			if (debug) {
				cout << "Performing row-wise multiplication" << endl;
			}			
		} else if (this->x == C_neighbor->x or this->x == C_neighbor->y) {
//		factor_col_i = factor_col_i (dot) message
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
					factor[dna_x][dna_y] *= messageFromNeighbor[dna_x];
				}
			}
			if (debug) {
				cout << "Performing column-wise multiplication" << endl;
			}			
		} else {
			cout << "Check product step" << endl;
            throw mt_error("check product step");
		}
	}	
	double scalingFactor = 0;
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			scalingFactor += factor[dna_x][dna_y];
		}
	}	
	if (scalingFactor == 0) {
        throw mt_error("check factor");
    }
	for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			this->belief[dna_x][dna_y] = factor[dna_x][dna_y]/scalingFactor;
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
					this->initialPotential[dna_p][dna_c] *= 0;
				} else {
					this->initialPotential[dna_p][dna_c] *= 1;
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
				this->initialPotential[dna_p][dna_c] *= x->rootProbability[dna_c];
			}
		}
	}
	double maxValue = 0;
	for (int i = 0; i < 4; i ++) {
		for (int j = 0; j < 4; j ++) {
			if (this->initialPotential[i][j] > maxValue) {
				maxValue = this->initialPotential[i][j];
			}
		}
	}
	this->belief = this->initialPotential;
	this->logScalingFactorForClique = 0;
	this->logScalingFactorForMessages.clear();
	this->messagesFromNeighbors.clear();
}

///...///...///...///...///...///...///...///... clique tree ...///...///...///...///...///...///...///...///

class cliqueTree {
public:
	vector < pair <clique *, clique *> > edgesForPreOrderTreeTraversal;
	vector < pair <clique *, clique *> > edgesForPostOrderTreeTraversal;
	vector < pair <clique *, clique *> > cliquePairsSortedWrtLengthOfShortestPath;
	map < pair <SEM_vertex *, SEM_vertex *>, Md> marginalizedProbabilitiesForVariablePair;
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
	Md GetP_XZ(SEM_vertex * X, SEM_vertex * Y, SEM_vertex * Z);	
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
		
		// Scope(Ci,Cl) = {X,Y}
		if (Ci->name < Cl->name) {
			tie(X,Y) = this->cliquePairToVariablePair[pair <clique*, clique*>(Ci,Cl)];
		} else {
			tie(Y,X) = this->cliquePairToVariablePair[pair <clique*, clique*>(Cl,Ci)];
		}			
				
				
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
			if (Y_temp != Y){
                throw mt_error("check Case 2b");
            }
		}
	}	
	
	return (tuple <SEM_vertex *,SEM_vertex *,SEM_vertex *>(X,Y,Z));
}

Md cliqueTree::GetP_XZ(SEM_vertex * X, SEM_vertex * Y, SEM_vertex * Z) {
	Md P_XY; Md P_YZ;
	Md P_ZGivenY; Md P_XZ;
	
    if (X->id < Y->id) {
		P_XY = this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(X,Y)];
	} else {		
		P_XY = MT(this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Y,X)]);
	}
	if (Y->id < Z->id) {
		P_YZ = this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Y,Z)];
	} else {		
		P_YZ = MT(this->marginalizedProbabilitiesForVariablePair[pair<SEM_vertex *, SEM_vertex *>(Z,Y)]);
	}
//	cout << "P_XY is " << endl << P_XY << endl;
//	cout << "P_YZ is " << endl << P_YZ << endl;
	P_ZGivenY = Md{};
	double rowSum;
	for (int row = 0; row < 4; row ++) {		
		rowSum = 0;
		for (int col = 0; col < 4; col ++) {
			rowSum += P_YZ[row][col];
		}
		for (int col = 0; col < 4; col ++) {
			if (rowSum != 0){
				P_ZGivenY[row][col] = P_YZ[row][col]/rowSum;
			}			
		}
	}
	
//	cout << "P_ZGivenY is " << endl << P_ZGivenY << endl;
	
	for (int row = 0; row < 4; row ++) {		
		for (int col = 0; col < 4; col ++) {
			P_XZ[row][col] = 0;
		}
	}
	
	for (int dna_y = 0; dna_y < 4; dna_y ++) {		
		for (int dna_x = 0; dna_x < 4; dna_x ++) {
			for (int dna_z = 0; dna_z < 4; dna_z ++) {					
				// Sum over Y
				P_XZ[dna_x][dna_z] += P_XY[dna_x][dna_y] * P_ZGivenY[dna_x][dna_z];
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
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Md>(pair<SEM_vertex *, SEM_vertex *>(C->x,C->y),C->belief));
		} else {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Md>(pair<SEM_vertex *, SEM_vertex *>(C->y,C->x),MT(C->belief)));
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

	Md P_XZ;
		
	//	Store P(X,Y) for each clique
	for (clique * C: this->cliques) {
		if (C->x->id < C->y->id) {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Md>(pair<SEM_vertex *, SEM_vertex *>(C->x,C->y),C->belief));
		} else {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Md>(pair<SEM_vertex *, SEM_vertex *>(C->y,C->x),MT(C->belief)));
			
		}
	}
	
	for (pair <clique *, clique *> cliquePair : this->cliquePairsSortedWrtLengthOfShortestPath) {
		tie (Ci, Cj) = cliquePair;
		tie (X, Y, Z) = this->GetXYZ(Ci, Cj);		
		this->cliquePairToVariablePair.insert(pair <pair <clique *, clique *>,pair <SEM_vertex *, SEM_vertex *>>(pair <clique *, clique *>(Ci,Cj), pair <SEM_vertex *, SEM_vertex *>(X,Z)));
		P_XZ = this->GetP_XZ(X, Y, Z);
		if (X->id < Z->id) {
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Md>(pair<SEM_vertex *, SEM_vertex *>(X,Z),P_XZ));			
		} else {			
			this->marginalizedProbabilitiesForVariablePair.insert(pair<pair<SEM_vertex *, SEM_vertex *>, Md>(pair<SEM_vertex *, SEM_vertex *>(Z,X),MT(P_XZ)));
		}
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
	double logScalingFactor;
	double largestElement;	
	array <double, 4> messageFromNeighbor;
	array <double, 4> messageToNeighbor;
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
	
	Md factor;
	factor = C_from->initialPotential;
	
	logScalingFactor = 0;
		// A. PRODUCT: Multiply messages from neighbors that are not C_to
	for (clique * C_neighbor : neighbors) {
		messageFromNeighbor = C_from->messagesFromNeighbors[C_neighbor];
		if (C_from->y == C_neighbor->x or C_from->y == C_neighbor->y) {
		// factor_row_i = factor_row_i (dot) message
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
				for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
					factor[dna_x][dna_y] *= messageFromNeighbor[dna_y];					
				}
			}
			if (verbose) {cout << "Performing row-wise multiplication" << endl;}			
		} else if (C_neighbor->x == C_from->x or C_neighbor->y == C_from->x) {
		// factor_col_i = factor_col_i (dot) message
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
					factor[dna_x][dna_y] *= messageFromNeighbor[dna_x];
				}
			}
			if (verbose) {cout << "Performing column-wise multiplication" << endl;}			
		} else {
			cout << "Check product step" << endl;
			throw mt_error("check product step");
		}		
		// Check to see if each entry in the factor is zero
		bool allZero = 1;
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				if (factor[dna_x][dna_y] == 0) {
					allZero = 0;
				}
			}
		}		
		// Rescale factor
		largestElement = 0;
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				if (largestElement < factor[dna_x][dna_y]) {
					largestElement = factor[dna_x][dna_y];
				}
			}
		}
		for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
			for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
				factor[dna_x][dna_y] /= largestElement;
			}
		}
		logScalingFactor += log(largestElement);
		logScalingFactor += C_from->logScalingFactorForMessages[C_neighbor];
	}
	// B. SUM
		// Marginalize factor by summing over common variable
	largestElement = 0;
	if (C_from->y == C_to->x or C_from->y == C_to->y) {
		// Sum over C_from->x		
		for (int dna_y = 0 ; dna_y < 4; dna_y ++) {
			messageToNeighbor[dna_y] = 0;
			for (int dna_x = 0 ; dna_x < 4; dna_x ++) {
				messageToNeighbor[dna_y] += factor[dna_x][dna_y];
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
				messageToNeighbor[dna_x] += factor[dna_x][dna_y];
			}
		}
		if (verbose) {
			cout << "Performing row-wise summation" << endl;
		}							
	} else {		
		cout << "Check sum step" << endl;
		throw mt_error("Check sum step");
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
			if(isnan(messageToNeighbor[dna])){
                throw mt_error("Check message to neighbor");
            }
		}
	}	
	// C. TRANSMIT
	C_to->logScalingFactorForMessages.insert(make_pair(C_from,logScalingFactor));
	C_to->messagesFromNeighbors.insert(make_pair(C_from,messageToNeighbor));
}

void cliqueTree::CalibrateTree() {
	clique * C_p; clique * C_c;
	
	//	Send messages from leaves to root	
	for (pair <clique *, clique *> cliquePair : this->edgesForPostOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		this->SendMessage(C_c, C_p);
	}

	//	Send messages from root to leaves	
	for (pair <clique *, clique *> cliquePair : this->edgesForPreOrderTreeTraversal) {
		tie (C_p, C_c) = cliquePair;
		this->SendMessage(C_p, C_c);
	}	
	//	Compute beliefs
	for (clique * C: this->cliques) {
		C->ComputeBelief();		
	}
}

///...///...///...///...///...///...///...///...///... SEM ...///...///...///...///...///...///...///...///...///

class SEM {
	
public:
	int largestIdOfVertexInMST = 1;
	default_random_engine generator;
	bool setParameters;
	bool verbose = 0;
	string modelForRooting;
	map <string,unsigned char> mapDNAtoInteger;
	map <int, SEM_vertex*> * vertexMap;
	vector <SEM_vertex*> vertices;
	map <pair<SEM_vertex *,SEM_vertex *>,Md> * M_hss;
	vector <int> sitePatternWeights;
	vector <vector <int> > sitePatternRepetitions;
	vector <int> sortedDeltaGCThresholds;
	int numberOfInputSequences;
	int numberOfVerticesInSubtree;
	int numberOfObservedVertices;
	int numberOfExternalVertices = 0;	
	int numberOfSitePatterns;
	int maxIter;
	double logLikelihoodConvergenceThreshold = 0.1;	
	double sumOfExpectedLogLikelihoods = 0;
	double maxSumOfExpectedLogLikelihoods = 0;
	int h_ind = 1;
	chrono::system_clock::time_point t_start_time;
	chrono::system_clock::time_point t_end_time;
	ofstream * logFile;
	SEM_vertex * root;
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForPostOrderTreeTraversal;
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForPreOrderTreeTraversal;	
	vector < pair <SEM_vertex *, SEM_vertex *>> edgesForChowLiuTree;
	vector < pair <SEM_vertex *, SEM_vertex *>> directedEdgeList;
	map <pair <SEM_vertex *, SEM_vertex *>, double> edgeLengths;
	vector < SEM_vertex *> leaves;
	vector < SEM_vertex *> preOrderVerticesWithoutLeaves;
	map < pair <SEM_vertex * , SEM_vertex *>, Md > expectedCountsForVertexPair;
	map < pair <SEM_vertex * , SEM_vertex *>, Md > posteriorProbabilityForVertexPair;
	map < SEM_vertex *, array <double,4>> expectedCountsForVertex; 
	map < SEM_vertex *, array <double,4>> posteriorProbabilityForVertex;
	map <int, Md> rateMatrixPerRateCategory;
	map <int, Md> rateMatrixPerRateCategory_stored;
	map <int, double> scalingFactorPerRateCategory;
	map <int, double> scalingFactorPerRateCategory_stored;
	int numberOfRateCategories = 0;
	double maximumLogLikelihood;
	Md I4by4;	
	cliqueTree * cliqueT;
	bool debug;
	bool finalIterationOfSEM;
	bool flag_logDet = 0;
	bool flag_Hamming = 0;
	bool flag_JC = 0;
	bool flag_added_duplicated_sequences = 0;
	map <string, int> nameToIdMap;
	string sequenceFileName;
	string phylip_file_name;
	string topologyFileName;
	string probabilityFileName;
	string probabilityFileName_best;
	string probabilityFileName_pars;
	string probabilityFileName_ssh;
	string probabilityFileName_diri;
	string prefix_for_output_files;
	string ancestralSequencesString = "";
	double sequenceLength;
	// Add vertices (and compressed sequence for leaves)
	array <double, 4> rootProbability;
	array <double, 4> rootProbability_stored;
	SEM_vertex * root_stored;
	vector <unsigned char> compressedSequenceToAddToMST;
	string nameOfSequenceToAddToMST;
	double logLikelihood;
	double logLikelihood_exp_counts;
	double logLikelihood_current;
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
	vector < tuple <string, string, double>> weightedEdgesToAddToGlobalPhylogeneticTree;
	vector < tuple <string, string, double>> edgeLogLikelihoodsToAddToGlobalPhylogeneticTree;		
	map <string, double> vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree;
	map <pair<SEM_vertex *,SEM_vertex *>,double> edgeLogLikelihoodsMap;
	SEM_vertex * externalVertex;
	void AddArc(SEM_vertex * from, SEM_vertex * to);
	void RemoveArc(SEM_vertex * from, SEM_vertex * to);
	void SetStream(ofstream& stream_to_set);
	void ClearDirectedEdges();
	void ClearUndirectedEdges();
	void ClearAllEdges();
	void AddVertex(string name, vector <unsigned char> compressedSequenceToAdd);
	void AddWeightedEdges(vector<tuple<string,string,double>> weightedEdgesToAdd);
	void AddEdgeLogLikelihoods(vector<tuple<string,string,double>> edgeLogLikelihoodsToAdd);
	void AddExpectedCountMatrices(map < pair <SEM_vertex * , SEM_vertex *>, Md > expectedCountsForVertexPair);
	void AddVertexLogLikelihoods(map <string,double> vertexLogLikelihoodsMapToAdd);
	void SetNumberOfVerticesInSubtree(int numberOfVertices);	
	void AddSitePatternWeights(vector <int> sitePatternWeightsToAdd);
	void AddSitePatternRepeats(vector <vector <int> > sitePatternRepetitionsToAdd);
	void AddSequences(vector <vector <unsigned char>> sequencesToAdd);	
	void OpenAncestralSequencesFile();
	void AddRootVertex();
	void SetVertexVector();
//	void AddCompressedSequencesAndNames(map<string,vector<unsigned char>> sequencesList, vector <vector <int>> sitePatternRepeats);
	void AddAllSequences(string sequencesFileName);
	void AddNames(vector <string> namesToAdd);
	void AddGlobalIds(vector <int> idsToAdd);
	// void ComputeNJTreeUsingHammingDistances(); 
	void ComputeNJTree_may_contain_uninitialized_values(); // Hamming (default) or LogDet distance
	void ComputeNJTree();
	double ComputeDistance(int v_i, int v_j);
	void RootedTreeAlongAnEdgeIncidentToCentralVertex();
	void RootTreeAlongAnEdgePickedAtRandom();
	void RootTreeAtAVertexPickedAtRandom();
	void RootTreeByFittingAGMMViaEM();
	void RootTreeByFittingUNREST();
	void RootTreeUsingSpecifiedModel(string modelForRooting);
	void RootTreeBySumOfExpectedLogLikelihoods();
	tuple<int,double,double,double,double> EM_started_with_SSH_parameters_rooted_at(SEM_vertex *v);
	tuple<int,double,double,double,double> EM_started_with_parsimony_rooted_at(SEM_vertex *v);
	tuple<int,double,double,double,double> EM_started_with_dirichlet_rooted_at(SEM_vertex *v);
	tuple<int,double,double,double,double> EM_root_search_with_parsimony_rooted_at(SEM_vertex *v);
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
	void WriteParametersOfGMM(string GMMparametersFileName);
	void RemoveEdgeLength(SEM_vertex * u, SEM_vertex * v);
	void AddEdgeLength(SEM_vertex * u, SEM_vertex * v, double t);
	double GetEdgeLength(SEM_vertex * u, SEM_vertex * v);
	double ComputeEdgeLength(SEM_vertex * u, SEM_vertex * v);
	void SetEdgeLength(SEM_vertex * u, SEM_vertex * v, double t);
	void SetEdgesFromTopologyFile();
	string EncodeAsDNA(vector<unsigned char> sequence);
	vector<unsigned char> DecompressSequence(vector<unsigned char>* compressedSequence, vector<vector<int>>* sitePatternRepeats);	
	void ComputeChowLiuTree();
	void AddSubforestOfInterest(SEM * localPhylogeneticTree);
	void ReadRootedTree(string treeFileName);
	void SetGMMparameters();
	void ReparameterizeGMM();
	void Set_pi_for_neighbors_of_root();
	array<double,4> get_pi_child();
	void ReadProbabilities();
	void WriteProbabilities(string fileName);
	void ReadTransitionProbabilities(string fileName);
	int GetVertexId(string v_name);
	SEM_vertex * GetVertex(string v_name);
	bool ContainsVertex(string v_name);
	int GetEdgeIndex (int vertexIndex1, int vertexIndex2, int numberOfVertices);
	Md GetP_yGivenx(Md P_xy);
	Md GetTransitionMatrix(SEM_vertex * p, SEM_vertex * c);
	array <double, 4> GetBaseComposition(SEM_vertex * v);
	array <double, 4> GetObservedCountsForVariable(SEM_vertex * v);
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
	void ComputeMLRootedTreeForRootSearchUnderGMM();	
	void ComputeMLEstimateOfGMMGivenExpectedDataCompletion();		
	void SetMinLengthOfEdges();
	void SetParametersForRateMatrixForNelderMead(double x[], int rateCat);
	void NelderMeadForOptimizingParametersForRateCat(int rateCat, int n, double start[], double xmin[], 
		 double *ynewlo, double reqmin, double step[], int konvge,
		 int kcount, int *icount, int *numres, int *ifault);
	void FitAGMModelViaHardEM();
	void ComputeInitialEstimateOfModelParameters();
	void SetInitialEstimateOfModelParametersUsingDirichlet();
	void SetInitialEstimateOfModelParametersUsingSSH();
	void TransformRootedTreeToBifurcatingTree();
	void SwapRoot();
	void SuppressRoot();
	bool IsTreeInCanonicalForm();
	bool root_search;
	string init_criterion;
	string parameter_file;
	void ComputeLogLikelihood();
	void ComputeLogLikelihoodUsingExpectedDataCompletion();
	pair <bool, SEM_vertex *> CheckAndRetrieveSingletonHiddenVertex();
	pair <bool, SEM_vertex *> CheckAndRetrieveHiddenVertexWithOutDegreeZeroAndInDegreeOne();
	pair <bool, SEM_vertex *> CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeOne();
	pair <bool, SEM_vertex *> CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeZero();
	pair <bool, SEM_vertex *> CheckAndRetrieveHiddenVertexWithOutDegreeGreaterThanTwo();
	pair <bool, SEM_vertex *> CheckAndRetrieveObservedVertexThatIsTheRoot();
	pair <bool, SEM_vertex *> CheckAndRetrieveObservedVertexThatIsNotALeafAndIsNotTheRoot();
	double GetExpectedMutualInformation(SEM_vertex * u, SEM_vertex * v);
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
	Md GetObservedCounts(SEM_vertex * u, SEM_vertex * v);
	void AddToExpectedCounts();
	void AddToExpectedCountsForEachVariable();
	void AddToExpectedCountsForEachVariablePair();
	Md GetExpectedCountsForVariablePair(SEM_vertex * u, SEM_vertex * v);
	Md GetPosteriorProbabilityForVariablePair(SEM_vertex * u, SEM_vertex * v);
	void AddToExpectedCountsForEachEdge();
	// Mutual information I(X;Y) is computing using 
	// P(X,Y), P(X), and P(Y), which in turn are computed using
	// A calibrated clique tree
	// using P(X,Y) = Sum_{H\{X,Y}}{P(X,Y|H\{X,Y},O)}
	// where H is the set of hidden variables and O is the set of observed variables	
	void RootTreeUsingEstimatedParametersViaML();
	void SetFlagForFinalIterationOfSEM();
	void OptimizeTopologyAndParametersOfGMM();
	int ConvertDNAtoIndex(char dna);	
	char GetDNAfromIndex(int dna_index);			
	double BIC;
	double AIC;
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
	void SetPrefixForOutputFiles(string prefix_for_output_files_to_set);
	void WriteRootedTreeInNewickFormat(string newickFileName);
	void WriteUnrootedTreeInNewickFormat(string newickFileName);
	void WriteCliqueTreeToFile(string cliqueTreeFileName);
	void WriteRootedTreeAsEdgeList(string fileName);
	void WriteUnrootedTreeAsEdgeList(string fileName);
	void ResetData();	
	void AddDuplicatedSequencesToRootedTree(MST * M);
	void AddDuplicatedSequencesToUnrootedTree(MST * M);
	void SetParameterFile();
	double EM_main(string init_criterion, bool root_search);
	void initialize_GMM(string init_criterion);
	
	double EM_rooted_at_each_internal_vertex_started_with_parsimony(int num_repetitions);
	double EM_rooted_at_each_internal_vertex_started_with_dirichlet(int num_repetitions);
	double EM_rooted_at_each_internal_vertex_started_with_SSH_par(int num_repetitions);
	void EM_root_search_at_each_internal_vertex_started_with_parsimony(int num_repetitions);
	void EM_root_search_at_each_internal_vertex_started_with_dirichlet(int num_repetitions);
	void EM_root_search_at_each_internal_vertex_started_with_SSH_par(int num_repetitions);
	// Select vertex for rooting Chow-Liu tree and update edges in T
	// Modify T such that T is a bifurcating tree and likelihood of updated
	// tree is equivalent to the likelihood of T
	SEM (int largestIdOfVertexInMST_toSet, double loglikelihood_conv_thresh, int max_EM_iter, bool verbose_flag_to_set) {
		this->root_search = false;
		this->logLikelihoodConvergenceThreshold = loglikelihood_conv_thresh;
		this->maxIter = max_EM_iter;
		this->distance_measure_for_NJ = "Hamming";		
		this->flag_Hamming = 1; this->flag_logDet = 0;
		this->verbose = verbose_flag_to_set;
		this->largestIdOfVertexInMST = largestIdOfVertexInMST_toSet;
		this->h_ind = 1;
		this->vertexMap = new map <int, SEM_vertex *> ;
		this->M_hss = new map <pair<SEM_vertex*,SEM_vertex*>,Md>;
		// this->vertexName2IdMap = new map <string, int> ;
		unsigned seed = chrono::system_clock::now().time_since_epoch().count();
		this->generator = default_random_engine(seed);
		this->I4by4 = Md{};
		for (int i = 0; i < 4; i++) {
			this->I4by4[i][i] = 1.0;
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
		delete this->M_hss;
	}
};

void SEM::SetStream(ofstream& stream_to_set){
	this->logFile = &stream_to_set;
}

void SEM::AddDuplicatedSequencesToRootedTree(MST * M) {
	// Store dupl seq names in uniq seq vertex
	double t;
	string uniq_seq_name;
	vector <string> dupl_seq_name_vec;
	SEM_vertex * u;
	SEM_vertex * p;
	SEM_vertex * d;
	SEM_vertex * h;
	vector <unsigned char> emptySequence;	
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
			d->AddParent(h);	
			h->AddChild(d);
			this->AddEdgeLength(h,d,0.0);			
		}
	}
	this->flag_added_duplicated_sequences = 1;
}

void SEM::AddDuplicatedSequencesToUnrootedTree(MST * M) {
	// Store dupl seq names in uniq seq vertex
	double t;
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
			d->AddNeighbor(h);	
			h->AddNeighbor(d);
			this->AddEdgeLength(h,d,0.0);			
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
			this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.insert(pair<string,double>(u->name,u->vertexLogLikelihood));
		}	
		if (this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.find(v->name) == this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.end()) {
			this->ComputeVertexLogLikelihood(v);
			this->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree.insert(pair<string,double>(v->name,v->vertexLogLikelihood));
		}
		this->ComputeEdgeLogLikelihood(u,v);
		this->ComputeEdgeLogLikelihood(v,u);
	}	
}

void SEM::ComputeVertexLogLikelihood(SEM_vertex * v) {
	array <double, 4> prob = this->posteriorProbabilityForVertex[v];
	array <double, 4> Counts = this->expectedCountsForVertex[v];
	v->vertexLogLikelihood = 0;
	for (int i = 0; i < 4; i ++) {
		if (prob[i] > 0) {
			v->vertexLogLikelihood += (log(prob[i]) * Counts[i]);
		}		
	}	
}


void SEM::ComputeEdgeLogLikelihood(SEM_vertex* x, SEM_vertex * y) {	
	Md P_xy = this->GetPosteriorProbabilityForVariablePair(x,y);
	Md P_yGivenx = this->GetP_yGivenx(P_xy);
	Md Counts = this->GetExpectedCountsForVariablePair(x,y);
	double edgeLogLikelihood = 0;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			if (P_yGivenx[dna_x][dna_y] > 0) {
				edgeLogLikelihood += (log(P_yGivenx[dna_x][dna_y]) * Counts[dna_x][dna_y]);
			}
		}
	}
	this->edgeLogLikelihoodsToAddToGlobalPhylogeneticTree.push_back(tuple <string, string, double>(x->name,y->name,edgeLogLikelihood));
}

void SEM::SetWeightedEdgesToAddToGlobalPhylogeneticTree() {
	this->weightedEdgesToAddToGlobalPhylogeneticTree.clear();
	int u_id; int v_id;
	SEM_vertex * u; SEM_vertex * v; 
	string u_name; string v_name;
	double t;	
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
	double n = this->sequenceLength;
	double numberOfFreeParameters = this->edgesForPostOrderTreeTraversal.size();
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
	if(this->vertexMap->size()!=1){
        throw mt_error("vertexMap not set correctly");
    }
	(*this->vertexMap)[-1]->compressedSequence.clear();	
}

SEM_vertex * SEM::GetVertex(string v_name){
	bool contains_v = this->ContainsVertex(v_name);
	SEM_vertex * node;
	int node_id;
	if(!contains_v) {
        throw mt_error("v_name not found");
    }
	
	node_id = this->GetVertexId(v_name);		
	node = (*this->vertexMap)[node_id];
	return node;

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

char SEM::GetDNAfromIndex(int dna_index){
	char bases[4] = {'A', 'C', 'G', 'T'};
	if (dna_index > -1 && dna_index <4){
		return bases[dna_index];
	} else {
		return 'Z';
	}
}

int SEM::ConvertDNAtoIndex(char dna){
	int value = -1;
	switch (dna)
	{
	case 'A':
		value = 0;
		break;
	case 'C':
		value = 1;
		break;
	case 'G':
		value = 2;
		break;
	case 'T':
		value = 3;
		break;
	default:
		value = -1;
        throw mt_error("DNA not valid");
		break;
	}	
	return (value);
}

void SEM::SetVertexVector(){
	this->vertices.clear();
	SEM_vertex * v;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		this->vertices.push_back(v);
	}
}


void SEM::SetGMMparameters() {
	this->ReadProbabilities();	
}

void SEM::ReparameterizeGMM() {
	
	// Compute pi, P(u,v) and P(v,u) for each vertex and edge using the method described in ssh paper	
	
	// 1. Set pi for root
	
	for (int p_id = 0; p_id < 4; ++p_id) {
		if(this->rootProbability[p_id] == 0) {throw mt_error("invalid probability");}
		this->root->root_prob_hss[p_id] = this->rootProbability[p_id];
	}

	// Store transition matrices in transition_prob_hss map, store root prob in node as root_prob_hss
	SEM_vertex * p; SEM_vertex * c; array <double,4> pi_p; array <double,4> pi_c;
	for (pair<SEM_vertex*, SEM_vertex*> edge : this->edgesForPreOrderTreeTraversal) {
		p = edge.first;
		c = edge.second;
		
		Md M_pc = c->transitionMatrix; // transition matrix of orig GMM parameters 
		Md M_cp; 					   // transition matrix of reparameterized GMM
		

		// 1. Store M_pc
		if(M_hss->find(edge) != M_hss->end()){
            throw mt_error("Check edges in preorder traversal");
        }
		(*this->M_hss)[{p,c}] = M_pc;

		// 2. Initialize pi_p and pi_c
		for (int x = 0; x < 4; x ++) pi_p[x] = p->root_prob_hss[x]; // root prob already computed
		for (int x = 0; x < 4; x ++) pi_c[x] = 0;					// root prob to be computed
		
		// 3. Compute pi_c
		for (int x = 0; x < 4 ; x ++){
			for (int y = 0; y < 4; y ++){
				pi_c[x] += pi_p[y] * M_pc[y][x];
			}
		}

		// 4. Store pi_c
		for (int x = 0; x < 4; x++) {
			c->root_prob_hss[x] = pi_c[x];
		}

		// 5. Compute M_cp for root at child		
		for (int x = 0; x < 4; x++) {
			for (int y = 0; y < 4; y++) {
				M_cp[y][x] = M_pc[x][y] * pi_p[x]/pi_c[y];			// Bayes rule as described in SSH paper
			}
		}
		
		// 6. Store M_cp
		(*this->M_hss)[{c,p}] = M_cp;
	}	
}

void SEM::ReadProbabilities() {
    std::ifstream inputFile(this->probabilityFileName_best);
    if (!inputFile) {
        throw mt_error("Failed to open probability file: " + this->probabilityFileName_best);
    }

    std::string line;
    // skip first two header lines
    if (!std::getline(inputFile, line) || !std::getline(inputFile, line)) {
        throw mt_error("Probability file too short (missing headers)");
    }

    std::string node_name;
	// string node_parent_name;
    double prob; int i, j;

    while (std::getline(inputFile, line)) {
        vector<string> splitLine = split_ws(line);
        const int num_words = static_cast<int>(splitLine.size());
		// cout << num_words << endl;
        switch (num_words) {
            case 8: { 
				// node_parent_name = splitLine[5];
                node_name = splitLine[7];
                break;
            }
            case 16: {                
                SEM_vertex* n = this->GetVertex(node_name);
                for (int p_id = 0; p_id < 16; ++p_id) {
                    i = p_id / 4;
                    j = p_id % 4;
                    try	{
						prob = stod(splitLine[p_id]); 
					}
					catch(const exception& e) {
						prob = 0;
						cout << "setting to 0 small prob value " << splitLine[p_id] << " not converted by stod" << endl;
						(*this->logFile) << "setting to 0 small prob value " << splitLine[p_id] << " not converted by stod" << endl;
					}
					n->transitionMatrix[i][j] = prob;
                }
                break;
            }
            case 9: {                
                node_name = splitLine[3];
                SEM_vertex* n = this->GetVertex(node_name);
                this->RootTreeAtVertex(n);
                break;
            }
            case 4: {                
                for (int p_id = 0; p_id < 4; ++p_id) {
                    prob = std::stod(splitLine[p_id]);
                    this->rootProbability[p_id] = prob;
                }
                break;
            }
            default:
                std::cerr << "ReadProbabilities: unexpected token count (" << num_words
                          << ") on line: " << line << "\n";
                break;
        }
    }
}

void SEM::WriteProbabilities(string fileName) {
	ofstream probabilityFile;
	probabilityFile.open(fileName);
	SEM_vertex * v;
	probabilityFile << "transition matrix for edge from " << "parent_name" << " to " << "child_name" << endl;
	
	for (int row = 0; row < 4; row++) {
		for (int col = 0; col < 4; col++) {
			probabilityFile << "p(" << this->GetDNAfromIndex(col) << "|" << this->GetDNAfromIndex(row) << ")";
			if (row ==3 and col ==3) {
				continue;
			} else {
				probabilityFile << " ";
			}
			
		}		
	}
	probabilityFile << endl;

	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		if (v != v->parent) {
			probabilityFile << "transition matrix for edge from " << v->parent->name << " to " << v->name << endl;
			for (int row = 0; row < 4; row++) {
				for (int col = 0; col < 4; col++) {
					probabilityFile << v->transitionMatrix[row][col];
					if (row ==3 and col ==3){
						continue;
					} else {
						probabilityFile << " ";
					}
				}				
			}
			probabilityFile << endl;
		}
	}

	probabilityFile << "probability at root " << this->root->name << " is ";
	for (int row = 0; row < 3; row++) {
		probabilityFile << "p(" << this->GetDNAfromIndex(row) << ") ";		
	}
	probabilityFile << "p(" << this->GetDNAfromIndex(3) << ") " << endl;

	for (int row = 0; row < 3; row++) {
		probabilityFile << this->rootProbability[row] << " ";
	}
	probabilityFile << this->rootProbability[3] << endl;
	probabilityFile.close();
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
		vector<string> splitLine = split_ws(line);
		u_name = splitLine[0];		
		v_name = splitLine[1];
		if (find(nonRootVertexNames.begin(),nonRootVertexNames.end(),v_name) == nonRootVertexNames.end()) {
			nonRootVertexNames.push_back(v_name);
		}		
		if (find(ancestorNames.begin(),ancestorNames.end(),u_name)==ancestorNames.end()) {
			ancestorNames.push_back(u_name);
		}
		if (find(leafNames.begin(),leafNames.end(),v_name)==leafNames.end()) {
			if(!starts_with(v_name, "h_")) {
				leafNames.push_back(v_name);
			}
		}
	}
	for (string name: leafNames) {
		SEM_vertex * v = new SEM_vertex(v_id, emptySequence);
		v->name = name;
		v->observed = 1;
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
	// Change root name
	
	ancestorNames.erase(remove(ancestorNames.begin(), ancestorNames.end(), rootName), ancestorNames.end());
	for (string name: ancestorNames) {
		SEM_vertex * v = new SEM_vertex(v_id,emptySequence);
		v->name = name;
		this->vertexMap->insert(pair <int,SEM_vertex*> (v_id,v));
		v_id += 1;
	}
	
	this->root = new SEM_vertex (((2 * n) - 2), emptySequence);	
	this->root->name = rootName;
	this->vertexMap->insert(pair <int,SEM_vertex*> (((2 * n) - 2), this->root));
	edgeListFile.clear();
	edgeListFile.seekg(0, ios::beg);
	for (string line; getline(edgeListFile, line);) {
		vector<string> split_ws(const string& s);
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
	// cout << "Number of leaves is " << this->leaves.size() << endl;
	this->SetEdgesForPreOrderTraversal();
	// cout << "Number of edges for pre order traversal is " << this->edgesForPreOrderTreeTraversal.size() << endl;
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
	// cout << "Number of vertices for pre order traversal is " << this->preOrderVerticesWithoutLeaves.size() << endl;
	this->SetEdgesForPostOrderTraversal();
	// cout << "Number of edges for post order traversal is " << this->edgesForPostOrderTreeTraversal.size() << endl;
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

void SEM::AddEdgeLength(SEM_vertex * u, SEM_vertex * v, double t) {	
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	this->edgeLengths[vertexPair] = t;
}

double SEM::GetEdgeLength(SEM_vertex * u, SEM_vertex * v) {
	double t;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (u->id < v->id) {
		vertexPair = make_pair(u,v);
	} else {
		vertexPair = make_pair(v,u);
	}
	t = this->edgeLengths[vertexPair];
	return (t);
}

double SEM::ComputeEdgeLength(SEM_vertex * u, SEM_vertex * v) {
	double t = 0;
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

void SEM::SetEdgeLength(SEM_vertex * u, SEM_vertex * v, double t) {
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
	double t;	
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
		if (v != this->externalVertex){
            throw mt_error("v should be equal to external vertex");
        }
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

Md SEM::GetTransitionMatrix(SEM_vertex * p, SEM_vertex * c) {	
	Md P = Md{};			
	int dna_p; int dna_c;
	for (int site = 0; site < this->numberOfSitePatterns; site ++) {
		if (p->compressedSequence[site] < 4 && c->compressedSequence[site] < 4) { // FIX_AMB
			dna_p = p->compressedSequence[site];
			dna_c = c->compressedSequence[site];		
			P[dna_p][dna_c] += this->sitePatternWeights[site];	
		}		
	}
//	cout << "Sequence of parent: " << EncodeAsDNA(p->compressedSequence) << endl;
//	cout << "Sequence of child: " << EncodeAsDNA(c->compressedSequence) << endl;
//	cout << "Count matrix is " << P << endl;
	double rowSum;
	for (int i = 0; i < 4; i ++) {
		rowSum = 0;
		for (int j = 0; j < 4; j ++) {
			rowSum += P[i][j];
		}
		for (int j = 0; j < 4; j ++) {
			 P[i][j] /= rowSum;
		}
	}
	return P;
}


void SEM::FitAGMModelViaHardEM() {
	this->ClearAncestralSequences();
	this->ComputeMPEstimateOfAncestralSequences();
	// Iterate till convergence of logLikelihood;
	double currentLogLikelihood = this->logLikelihood;
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

void SEM::SetPrefixForOutputFiles(string prefix_for_output_files_to_set){
	this->prefix_for_output_files = prefix_for_output_files_to_set;
}

void SEM::WriteRootedTreeInNewickFormat(string newickFileName) {
	vector <SEM_vertex *> verticesToVisit;
	SEM_vertex * c;
	SEM_vertex * p;	
	double edgeLength;	
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
	double t;
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


void SEM::WriteParametersOfGMM(string fileName) {
	ofstream parameterFile;
	parameterFile.open(fileName);		
	
	parameterFile << "Root probability for vertex " << this->root->name << " is " << endl;
	for (int i = 0; i < 4; i++) {
		parameterFile << this->rootProbability[i] << "\t";
	}
	parameterFile << endl;


	SEM_vertex * c; SEM_vertex * p;
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap){
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			c->transitionMatrix = this->GetTransitionMatrix(p,c);					
			parameterFile << "Transition matrix for " << p->name << " to " << c->name << " is " << endl;
			string trans_par_string = "";
			for (int i = 0; i < 4; i++) {
				for (int j = 0; j < 4; j++) {
					trans_par_string.append(to_string(c->transitionMatrix[i][j]) + " ");					
				}
			}
			if (!trans_par_string.empty() && trans_par_string.back() == ' ') trans_par_string.pop_back();    		
			parameterFile << trans_par_string << endl;
		}
	}	
	parameterFile.close();
}

void SEM::WriteRootedTreeAsEdgeList(string fileName) {
	ofstream treeFile;
	treeFile.open(fileName);
	double t;
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
	double siteWeight = this->sitePatternWeights[this->cliqueT->site];	
	// Add to counts for each unobserved vertex (C->x) where C is a clique
	array <double, 4> marginalizedProbability;
	vector <SEM_vertex *> vertexList;
	for (clique * C: this->cliqueT->cliques) {
		v = C->x;
		if(v->observed){
            throw mt_error("v should not be am observed vertex");
        }
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
	double siteWeight = this->sitePatternWeights[this->cliqueT->site];	
	pair <SEM_vertex *, SEM_vertex*> vertexPair;
	Md countMatrixPerSite;
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
							this->expectedCountsForVertexPair[vertexPair][dna_u][dna_v] += countMatrixPerSite[dna_u][dna_v] * siteWeight;
						}
					}
				}
			}
		}
	}
}

void SEM::AddExpectedCountMatrices(map < pair <SEM_vertex * , SEM_vertex *>, Md > expectedCountsForVertexPairToAdd) {
	string u_name;
	string v_name;
	SEM_vertex * u;
	SEM_vertex * v;
	pair <SEM_vertex *, SEM_vertex *> edge;
	Md CountMatrix;
	for (pair <pair <SEM_vertex * , SEM_vertex *>, Md> mapElem: expectedCountsForVertexPairToAdd) {
		u_name = mapElem.first.first->name;
		v_name = mapElem.first.second->name;
		CountMatrix = mapElem.second;
		u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		if (u->id < v->id) {
			this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(u,v)] = CountMatrix;
		} else {			
			this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(v,u)] = MT(CountMatrix);

		}
	}	//this->expectedCountsForVertexPair
}


Md SEM::GetExpectedCountsForVariablePair(SEM_vertex * u, SEM_vertex * v) {
	Md C_pc;
	if (u->id < v->id) {
		C_pc = this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(u,v)];	
	} else {		
		C_pc = MT(this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(v,u)]);
	}
	return (C_pc);
}

Md SEM::GetPosteriorProbabilityForVariablePair(SEM_vertex * u, SEM_vertex * v) {
	Md P;
	if (u->id < v->id) {
		P = this->posteriorProbabilityForVertexPair[pair<SEM_vertex *, SEM_vertex *>(u,v)];
	} else {		
		P = MT(this->posteriorProbabilityForVertexPair[pair<SEM_vertex *, SEM_vertex *>(v,u)]);
	}
	return (P);
}

void SEM::AddToExpectedCountsForEachEdge() {
	double siteWeight = this->sitePatternWeights[this->cliqueT->site];
	Md countMatrixPerSite;
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
				this->expectedCountsForVertexPair[vertexPair][dna_u][dna_v] += countMatrixPerSite[dna_u][dna_v] * siteWeight;
			}
		}
	}
}

void SEM::AddToExpectedCounts() {
	SEM_vertex * u; SEM_vertex * v;
	double siteWeight = this->sitePatternWeights[this->cliqueT->site];	
	// Add to counts for each unobserved vertex (C->x) where C is a clique
	array <double, 4> marginalizedProbability;
	vector <SEM_vertex *> vertexList;
	for (clique * C: this->cliqueT->cliques) {
		v = C->x;
		if(v->observed){
            throw mt_error("v should not be observed");
        }
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
	Md countMatrixPerSite;
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
							this->expectedCountsForVertexPair[vertexPair][dna_u][dna_v] += countMatrixPerSite[dna_u][dna_v] * siteWeight;
						}
					}
				}
			}
		}
	}
}

Md SEM::GetObservedCounts(SEM_vertex * u, SEM_vertex * v) {	
	Md countMatrix = Md{};
	int dna_u; int dna_v;
	for (int i = 0; i < this->sequenceLength; i++) {
		dna_u = u->compressedSequence[i];
		dna_v = v->compressedSequence[i];
		countMatrix[dna_u][dna_v] += this->sitePatternWeights[i];
	}
	return (countMatrix);
}

void SEM::ComputeExpectedCountsForRootSearch() {
//	cout << "Initializing expected counts" << endl;
	this->InitializeExpectedCountsForEachVariable();
	this->InitializeExpectedCountsForEachEdge();
//	this->ResetExpectedCounts();
//	SEM_vertex * x; SEM_vertex * y; 
	Md P_XY;
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
	array <double, 4> posteriorProbability;
	int maxProbState;
	double maxProb;
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
				if(maxProbState == -1) {
                    throw mt_error("Check prob assignment");
                }
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
	Md P_XY;
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
		// if (debug) {
		// 	cout << "Number of post prob is " << this->cliqueT->marginalizedProbabilitiesForVariablePair.size() << endl; 
		// 	for (pair < pair <SEM_vertex *, SEM_vertex *>, Md> vertexPairToMatrix : this->cliqueT->marginalizedProbabilitiesForVariablePair) {
		// 		tie (x, y) = vertexPairToMatrix.first;
		// 		P_XY = vertexPairToMatrix.second;		
		// 		cout << "P(X,Y) is " << endl;
		// 		cout << P_XY << endl;			
		// 		dna_x = x->compressedSequence[site];
		// 		dna_y = y->compressedSequence[site];
		// 		cout << "dna_x is " << dna_x;
		// 		cout << " and " << "dna_y is " << dna_y << endl;			
		// 		cout << "P(" << x->name << "," << y->name << ") for site " << site;  
		// 		cout << " is " << P_XY[dna_x][dna_y] << endl;
		// 	}			
		// }
		this->AddToExpectedCountsForEachVariable();
		this->AddToExpectedCountsForEachVariablePair();
		// if (debug) {
		// 	break;
		// }
	}
	// Compare observed counts with expected counts (done)
	if (debug) {
		Md observedCounts;
		Md expectedCounts;
		for (pair <pair<SEM_vertex *, SEM_vertex *>, Md> vertexPairToCountMatrix : this->expectedCountsForVertexPair) {
			tie (x, y) = vertexPairToCountMatrix.first;
			expectedCounts = vertexPairToCountMatrix.second;
			observedCounts = this->GetObservedCounts(x,y);
			// cout << "Observed count matrix is " << endl;
			// cout << observedCounts << endl;
			// cout << "Expected count matrix is " << endl;
			// cout << expectedCounts << endl;
			// cout << "======================" << endl;
		}
	}
}

void SEM::ComputePosteriorProbabilitiesUsingExpectedCounts() {	
	SEM_vertex * v;
	double sum;
	// Compute posterior probability for vertex
	this->posteriorProbabilityForVertex.clear();
	array <double, 4> P_X;	
	for (pair <SEM_vertex *, array <double, 4>> vertexAndCountArray: this->expectedCountsForVertex) {
		v = vertexAndCountArray.first;
		P_X = vertexAndCountArray.second;
		sum = 0;
		for (int i = 0; i < 4; i++) {
			sum += P_X[i];
		}
		for (int i = 0; i < 4; i++) {
			P_X[i] /= sum;
		}
		this->posteriorProbabilityForVertex.insert(pair<SEM_vertex * , array <double, 4>>(v,P_X));
	}
	// Compute posterior probability for vertex pair
	this->posteriorProbabilityForVertexPair.clear();
	Md P_XY;
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	for (pair <pair<SEM_vertex *, SEM_vertex *>, Md> vertexPairAndCountMatrix: this->expectedCountsForVertexPair) {
		vertexPair = vertexPairAndCountMatrix.first;
		P_XY = vertexPairAndCountMatrix.second;
		sum = 0;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				sum += P_XY[i][j];
			}
		}
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				P_XY[i][j] /= sum;
			}
		}
		this->posteriorProbabilityForVertexPair.insert(pair<pair<SEM_vertex *, SEM_vertex *>,Md>(vertexPair,P_XY));
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
                    throw mt_error("Check root of clique tree");
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
					this->expectedCountsForVertexPair[pair <SEM_vertex *, SEM_vertex *>(u,v)] = Md{};
				}	
			}			
		}
	}
}

void SEM::InitializeExpectedCountsForEachVariable() {
	SEM_vertex * v;
	// Initialize expected counts for each vertex
	this->expectedCountsForVertex.clear();
	array <double, 4> observedCounts;
	for (pair<int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		for (int i = 0; i < 4; i++) {
			observedCounts[i] = 0;
		}
		if (v->observed) {			
			observedCounts = this->GetObservedCountsForVariable(v);
		}
		this->expectedCountsForVertex.insert(pair<SEM_vertex *, array<double,4>>(v,observedCounts));
	}	
}

void SEM::InitializeExpectedCountsForEachVariablePair() {
	SEM_vertex * u; SEM_vertex * v;
	// Initialize expected counts for each vertex pair
	this->expectedCountsForVertexPair.clear();	
	Md countMatrix;	
	int dna_u;
	int dna_v;
	for (pair<int,SEM_vertex *> idPtrPair_1 : * this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex *> idPtrPair_2 : * this->vertexMap) {
			v = idPtrPair_2.second;
			if (u->id < v->id) {
				countMatrix = Md{};			
				if (u->observed and v->observed) {
					for (int site = 0; site < this->numberOfSitePatterns; site++) {
						dna_u = u->compressedSequence[site];
						dna_v = v->compressedSequence[site];
						if (dna_u < 4 && dna_v < 4) { // FIX_AMB
							countMatrix[dna_u][dna_v] += this->sitePatternWeights[site];
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
	Md countMatrix;
	for (pair <SEM_vertex *, SEM_vertex *> edge : this->edgesForPreOrderTreeTraversal) {
		countMatrix = Md{};
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
	array <double, 4> observedCounts;
	for (pair<int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		v = idPtrPair.second;
		for (int i = 0; i < 4; i++) {
			observedCounts[i] = 0;
		}
		if (v->observed) {			
			observedCounts = this->GetObservedCountsForVariable(v);
		}
		this->expectedCountsForVertex.insert(pair<SEM_vertex *, array<double,4>>(v,observedCounts));
	}
	
	// Initialize expected counts for each vertex pair
	this->expectedCountsForVertexPair.clear();	
	Md countMatrix;	
	int dna_u;
	int dna_v;
	for (pair<int,SEM_vertex *> idPtrPair_1 : * this->vertexMap) {
		u = idPtrPair_1.second;
		for (pair<int,SEM_vertex *> idPtrPair_2 : * this->vertexMap) {
			v = idPtrPair_2.second;
			if (u->id < v->id) {
				countMatrix = Md{};			
				if (u->observed and v->observed) {
					for (int site = 0; site < this->numberOfSitePatterns; site++) {
						dna_u = u->compressedSequence[site];
						dna_v = v->compressedSequence[site];
						countMatrix[dna_u][dna_v] += this->sitePatternWeights[site];
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


void SEM::ComputeLogLikelihoodUsingExpectedDataCompletion() {
	this->logLikelihood = 0;
	array <double, 4> S_r = this->expectedCountsForVertex[this->root];
	for (int dna_r = 0; dna_r < 4; dna_r ++) {
		if (this->rootProbability[dna_r] > 0) {
			this->logLikelihood += S_r[dna_r] * log(this->rootProbability[dna_r]);
		}	
	}
	// Contribution of edges
	SEM_vertex * p; SEM_vertex * c;
	Md S_pc;
	Md P;
	for (pair <int, SEM_vertex *> idPtrPair : *this->vertexMap) {
		c = idPtrPair.second;
		p = c->parent;
		if (p != c) {
			P = c->transitionMatrix;
			if (p->id < c->id) {
				S_pc = this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(p,c)];	
			} else {				
				S_pc = MT(this->expectedCountsForVertexPair[pair<SEM_vertex*,SEM_vertex*>(c,p)]);
			}
			for (int dna_p = 0; dna_p < 4; dna_p ++) {
				for (int dna_c = 0; dna_c < 4; dna_c ++) {
					if(S_pc[dna_p][dna_c] < 0) {
						cout << "Expected counts for " << p->name << "\t" << c->name << " is " << endl;						
						throw mt_error("Check counts");
					}
					if (P[dna_p][dna_c] > 0) {
						this->logLikelihood += S_pc[dna_p][dna_c] * log(P[dna_p][dna_c]);
					}
				}
			}
		}
	}
}

// Case 1: Observed vertices may have out degree > 0
// Case 2: Root may have out degree = one
// Case 3: Directed tree (rooted) with vertices with outdegree 2 or 0.
void SEM::ComputeLogLikelihood() {
	this->logLikelihood = 0;
	map <SEM_vertex*,array<double,4>> conditionalLikelihoodMap;
	std::array <double,4> conditionalLikelihood;
	double partialLikelihood;
	double siteLikelihood;	
	double largestConditionalLikelihood = 0;
	double currentProb;			
	vector <SEM_vertex *> verticesToVisit;	
	SEM_vertex * p;
	SEM_vertex * c;
	Md P;
	for (int site = 0; site < this->numberOfSitePatterns; site++) {
		conditionalLikelihoodMap.clear();
		this->ResetLogScalingFactors();
		for (pair<SEM_vertex *,SEM_vertex *> edge : this->edgesForPostOrderTreeTraversal){
			tie (p, c) = edge;
			P = c->transitionMatrix;
			p->logScalingFactors += c->logScalingFactors;
			// Initialize conditional likelihood for leaves
			if (c->outDegree==0) {
				for (unsigned char dna_c = 0; dna_c < 4; dna_c ++) {
					conditionalLikelihood[dna_c] = 0;
				}
				conditionalLikelihood[c->compressedSequence[site]] = 1;
				conditionalLikelihoodMap.insert(pair <SEM_vertex *,array<double,4>>(c,conditionalLikelihood));
			}
			// Initialize conditional likelihood for ancestors
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()) {
				// Case 1: Ancestor is not an observed vertex
				if (p->id > this->numberOfObservedVertices -1) {
					for (unsigned char dna_c = 0; dna_c < 4; dna_c++){
						conditionalLikelihood[dna_c] = 1;
					}
				} else {
				// Case 2: Ancestor is an observed vertex
					for (unsigned char dna_c = 0; dna_c < 4; dna_c ++) {
						conditionalLikelihood[dna_c] = 0;
					}
					conditionalLikelihood[p->compressedSequence[site]] = 1;
				}								
				conditionalLikelihoodMap.insert(pair <SEM_vertex *,array<double,4>>(p,conditionalLikelihood));					
			}			
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()) {
				for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
				conditionalLikelihood[dna_c] = 1;
				}
				conditionalLikelihoodMap.insert(pair <SEM_vertex *,array<double,4>>(p,conditionalLikelihood));
			}
			largestConditionalLikelihood = 0;
			for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
				partialLikelihood = 0;
				for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
//					if (P[dna_p][dna_c]*conditionalLikelihoodMap[c][dna_c] == 0 and P[dna_p][dna_c] > 0 and conditionalLikelihoodMap[c][dna_c] > 0) {
//						cout << "Numerical underflow in computing partial likelihood" << endl;
//						cout << "P(y|x) is " << P[dna_p][dna_c] << endl;
//						cout << "L(y) is " << conditionalLikelihoodMap[c][dna_c] << endl;
//						cout << "2^-256 is " << 1.0/pow(2,256) << endl;
//					}
					partialLikelihood += P[dna_p][dna_c]*conditionalLikelihoodMap[c][dna_c];
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
				throw mt_error("Largest conditional likelihood value is zero");
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

void SEM::OptimizeTopologyAndParametersOfGMM() {
	double logLikelihood_current;
	int iter = 0;		
	bool continueIterations = 1;	
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
	if (nH != nL-1) {
        throw mt_error("Graph is not a tree");
    }
//		cout << "Initial estimate of ancestral sequences via MP" << endl;
	this->ComputeMPEstimateOfAncestralSequences();
//		cout << "Initial estimate of model parameters for fully labeled tree" << endl;
	this->ComputeInitialEstimateOfModelParameters();
	this->ClearAncestralSequences();
	this->ComputeLogLikelihood();
	cout << "Log-likelihood computed with parameters initialized using parsimony is " << this->logLikelihood << endl;
	logLikelihood_current = this->logLikelihood;
	while (continueIterations) {
		iter += 1;
// 1. Construct clique tree				
		
		this->ConstructCliqueTree();
// this->WriteCliqueTreeToFile(cliqueTreeFileName);
		// 2. Compute expected counts		

		this->ComputeExpectedCountsForFullStructureSearch();
		// Compare expected counts with actual counts
		// 3. Compute posterior probabilities using expected counts
		
		this->ComputePosteriorProbabilitiesUsingExpectedCounts();
		// 4. Compute Chow-Liu tree
		
		this->ComputeChowLiuTree();
//			this->WriteUnrootedTreeAsEdgeList(chowLiuTreeFileName);
		// 5. Transform to ML tree s.t. each the out degree of each vertex is either zero or two.
		
		this->ComputeMLRootedTreeForFullStructureSearch();
//			this->WriteRootedTreeAsEdgeList(MLRootedTreeFileName);
		// 6. Repeat steps 1 through 6 till convergence		
		
		// cout << "Loglikelihood before iteration " << iter << " of structural EM is " << logLikelihood_current + this->logLikelihoodConvergenceThreshold << endl;
		// cout << "Loglikelihood after iteration " << iter << " of structural EM is " << this->logLikelihood << endl;	
										
		if ((this->logLikelihood > logLikelihood_current + this->logLikelihoodConvergenceThreshold) and (iter < this->maxIter)) {
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

void SEM::EM_root_search_at_each_internal_vertex_started_with_parsimony(int num_repetitions) {
	// cout << "convergence threshold for EM is " << this->logLikelihoodConvergenceThreshold << endl;
	// (* this->logFile) << "convergence threshold for EM is " << this->logLikelihoodConvergenceThreshold << endl;
	// cout << "maximum number of EM iterations allowed is " << this->maxIter << endl;
	// (* this->logFile) << "maximum number of EM iterations allowed is " << this->maxIter << endl;
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
	SEM_vertex * v;
	vector <double> loglikelihoodscoresForEachRepetition;
	ofstream loglikelihood_node_rep_file;
	loglikelihood_node_rep_file.open(this->prefix_for_output_files + ".rooting_initial_final_rep_loglik");
	double max_log_likelihood = -1 * pow(10,5);
	double logLikelihood_pars;
	double loglikelihood_edc_first;
	double loglikelihood_edc_final;
	double logLikelihood_final;
	int iter;	
	tuple <int,double,double,double,double> iter_parsll_edllfirst_edllfinal_llfinal;
	vector<int> vertex_indices_to_visit;
	
	for (int v_ind = n; v_ind < num_vertices; v_ind++) {
		vertex_indices_to_visit.push_back(v_ind);
	}
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);

	for (int v_ind : vertex_indices_to_visit) {
		v = (*this->vertexMap)[v_ind];
		// cout << v->name << endl;
		if(v->degree != 3) {
            throw mt_error("Check input topology. Expects internal nodes to have degree 3");
        }
		loglikelihoodscoresForEachRepetition.clear();
		for (int rep = 0; rep < num_repetitions; rep++) {					
			iter_parsll_edllfirst_edllfinal_llfinal = this->EM_root_search_with_parsimony_rooted_at(v);			
			iter = get<0>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_pars = get<1>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_edc_first = get<2>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_edc_final = get<3>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_final = get<4>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_node_rep_file << v->name << "\t"
										<< this->root->name << "\t"
										<< rep +1 << "\t"
										<< iter << "\t"
										<< setprecision(ll_precision) << logLikelihood_pars << "\t"
										<< setprecision(ll_precision) << loglikelihood_edc_first << "\t"
										<< setprecision(ll_precision) << loglikelihood_edc_final << "\t"
										<< setprecision(ll_precision) << logLikelihood_final << endl;
			if (max_log_likelihood < logLikelihood_final) {
				this->WriteProbabilities(this->prefix_for_output_files + ".probability_max_ll_pars");
				max_log_likelihood = logLikelihood_final;
			}
		}
	}
	
	loglikelihood_node_rep_file.close();	
	//  cout << "max log-likelihood is " << setprecision(ll_precision) << max_log_likelihood << endl;
	// (*this->logFile) << "max log-likelihood is " << setprecision(ll_precision) << max_log_likelihood << endl;
}




vector<unsigned char> SEM::DecompressSequence(vector<unsigned char>* compressedSequence, vector<vector<int>>* sitePatternRepeats){
	int totalSequenceLength = 0;
	for (vector<int> sitePatternRepeat: *sitePatternRepeats){
		totalSequenceLength += int(sitePatternRepeat.size());
	}
	vector <unsigned char> decompressedSequence;
	for (int v_ind = 0; v_ind < totalSequenceLength; v_ind++){
		decompressedSequence.push_back(char(0));
	}
	unsigned char dnaToAdd;
	for (int sitePatternIndex = 0; sitePatternIndex < int(compressedSequence->size()); sitePatternIndex++){
		dnaToAdd = (*compressedSequence)[sitePatternIndex];
		for (int pos: (*sitePatternRepeats)[sitePatternIndex]){
			decompressedSequence[pos] = dnaToAdd;
		}
	}
	return (decompressedSequence);	
}

string SEM::EncodeAsDNA(vector<unsigned char> sequence){
	string allDNA = "AGTC";
	string dnaSequence = "";
	for (unsigned char s : sequence){
		dnaSequence += allDNA[s];
	}
	return dnaSequence;
}

void SEM::initialize_GMM(string init_criterion) {
	if (init_criterion == "ssh"){

	} else if (init_criterion == "parsimony"){

	} else if (init_criterion == "dirichlet") {

	} else {
		throw mt_error("initialization criterion not recognized");
	}
}

void SEM::SetParameterFile(){
	// if (this->init_criterion == "ssh") {
	// 	this->parameterFile;		

	// } else if (this->init_criterion == "parsimony") {

	// } else if (init_criterion == "dirichlet") {
		
	// } else {
	// 	throw mt_error("initialization criterion not recognized");
	// }
}

double SEM::EM_rooted_at_each_internal_vertex_started_with_dirichlet(int num_repetitions) {
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
	SEM_vertex * v;
	vector <double> loglikelihoodscoresForEachRepetition;
	ofstream loglikelihood_node_rep_file;
	loglikelihood_node_rep_file.open(this->prefix_for_output_files + ".dirichlet_rooting_initial_final_rep_loglik");
    loglikelihood_node_rep_file << "root" << "\t"										
                                << "rep" << "\t"
                                << "iter" << "\t"
                                << "ll dirichlet" << "\t"
                                << "ecd-ll first" << "\t"
                                << "ecd-ll final" << "\t"
                                << "ll final" << endl;
	double max_log_likelihood = -1 * pow(10,5);
	double logLikelihood_pars;
	double loglikelihood_edc_first;
	double loglikelihood_edc_final;
	double logLikelihood_final;
	int iter;	
	tuple <int,double,double,double,double> iter_dirill_edllfirst_edllfinal_llfinal;
	vector<int> vertex_indices_to_visit;
	
	for (int v_ind = n; v_ind < num_vertices; v_ind++) {
		vertex_indices_to_visit.push_back(v_ind);
	}
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);
    int v_ind;
	for (int v_i = 0; v_i < vertex_indices_to_visit.size(); v_i++){
        v_ind = vertex_indices_to_visit[v_i];
		v = (*this->vertexMap)[v_ind];
		cout << "node " << v_i+1 << ":" << v->name << endl;
		if(v->degree != 3){
            throw mt_error("Expect internal nodes to have degree three");
        }
		loglikelihoodscoresForEachRepetition.clear();
		for (int rep = 0; rep < num_repetitions; rep++) {					
			iter_dirill_edllfirst_edllfinal_llfinal = this->EM_started_with_dirichlet_rooted_at(v);			
			iter = get<0>(iter_dirill_edllfirst_edllfinal_llfinal);
			logLikelihood_pars = get<1>(iter_dirill_edllfirst_edllfinal_llfinal);
			loglikelihood_edc_first = get<2>(iter_dirill_edllfirst_edllfinal_llfinal);
			loglikelihood_edc_final = get<3>(iter_dirill_edllfirst_edllfinal_llfinal);
			logLikelihood_final = get<4>(iter_dirill_edllfirst_edllfinal_llfinal);
			loglikelihood_node_rep_file << v->name << "\t"										
										<< rep +1 << "\t"
										<< iter << "\t"
										<< setprecision(ll_precision) << logLikelihood_pars << "\t"
										<< setprecision(ll_precision) << loglikelihood_edc_first << "\t"
										<< setprecision(ll_precision) << loglikelihood_edc_final << "\t"
										<< setprecision(ll_precision) << logLikelihood_final << endl;
			if (max_log_likelihood < logLikelihood_final) {                
				this->WriteProbabilities(this->probabilityFileName_diri);
				max_log_likelihood = logLikelihood_final;
			}
		}
	}
	
	loglikelihood_node_rep_file.close();	
	 cout << "max log likelihood obtained using Dirichlet parameters is " << setprecision(ll_precision) << max_log_likelihood << endl;
	(*this->logFile) << "max log likelihood obtained using Dirichlet parameters is " << setprecision(ll_precision) << max_log_likelihood << endl;
	return max_log_likelihood;
}

double SEM::EM_rooted_at_each_internal_vertex_started_with_parsimony(int num_repetitions) {
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
	SEM_vertex * v;
	vector <double> loglikelihoodscoresForEachRepetition;
	ofstream loglikelihood_node_rep_file;
	loglikelihood_node_rep_file.open(this->prefix_for_output_files + ".pars_rooting_initial_final_rep_loglik");
    loglikelihood_node_rep_file << "root" << "\t"										
                                << "rep" << "\t"
                                << "iter" << "\t"
                                << "ll pars" << "\t"
                                << "edc-ll first" << "\t"
                                << "edc-ll final" << "\t"
                                << "ll final" << endl;
	double max_log_likelihood = -1 * pow(10,5);
	double logLikelihood_pars;
	double loglikelihood_edc_first;
	double loglikelihood_edc_final;
	double logLikelihood_final;
	int iter;	
	tuple <int,double,double,double,double> iter_parsll_edllfirst_edllfinal_llfinal;
	vector<int> vertex_indices_to_visit;
	
	for (int v_ind = n; v_ind < num_vertices; v_ind++) {
		vertex_indices_to_visit.push_back(v_ind);
	}
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);
    int v_ind;
	for (int v_i = 0; v_i < vertex_indices_to_visit.size(); v_i++){
        v_ind = vertex_indices_to_visit[v_i];
		v = (*this->vertexMap)[v_ind];
		cout << "node " << v_i+1 << ":" << v->name << endl;
		if(v->degree != 3){
            throw mt_error("Expect internal nodes to have degree three");
        }
		loglikelihoodscoresForEachRepetition.clear();
		for (int rep = 0; rep < num_repetitions; rep++) {					
			iter_parsll_edllfirst_edllfinal_llfinal = this->EM_started_with_parsimony_rooted_at(v);			
			iter = get<0>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_pars = get<1>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_edc_first = get<2>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_edc_final = get<3>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_final = get<4>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_node_rep_file << v->name << "\t"										
										<< rep +1 << "\t"
										<< iter << "\t"
										<< setprecision(ll_precision) << logLikelihood_pars << "\t"
										<< setprecision(ll_precision) << loglikelihood_edc_first << "\t"
										<< setprecision(ll_precision) << loglikelihood_edc_final << "\t"
										<< setprecision(ll_precision) << logLikelihood_final << endl;
			if (max_log_likelihood < logLikelihood_final) {                
				this->WriteProbabilities(this->probabilityFileName_pars);
				max_log_likelihood = logLikelihood_final;
			}
		}
	}
	
	loglikelihood_node_rep_file.close();	
	 cout << "max log likelihood obtained using Parsimony parameters is " << setprecision(ll_precision) << max_log_likelihood << endl;
	(*this->logFile) << "max log likelihood obtained using Parsimony parameters is " << setprecision(ll_precision) << max_log_likelihood << endl;
	return max_log_likelihood;	
}

double SEM::EM_rooted_at_each_internal_vertex_started_with_SSH_par(int num_repetitions) {
	// cout << "convergence threshold for EM is " << this->logLikelihoodConvergenceThreshold << endl;
	// (* this->logFile) << "convergence threshold for EM is " << this->logLikelihoodConvergenceThreshold << endl;
	// cout << "maximum number of EM iterations allowed is " << this->maxIter << endl;
	// (* this->logFile) << "maximum number of EM iterations allowed is " << this->maxIter << endl;
	int n = this->numberOfObservedVertices;
	int num_vertices = this->vertexMap->size();	
	SEM_vertex * v;
	vector <double> loglikelihoodscoresForEachRepetition;
	ofstream loglikelihood_node_rep_file;
	loglikelihood_node_rep_file.open(this->prefix_for_output_files + ".SSH_rooting_initial_final_rep_loglik");
    loglikelihood_node_rep_file << "root" << "\t"
                                << "rep" << "\t"
                                << "iter" << "\t"
                                << "ll SSH" << "\t"
                                << "ecd ll first" << "\t"
                                << "ecd ll final" << "\t"
                                << "ll final" << endl;
	double max_log_likelihood = -1 * pow(10,5);	
	double logLikelihood_pars;
	double loglikelihood_edc_first;
	double loglikelihood_edc_final;
	double logLikelihood_final;
	int iter;	
	tuple <int,double,double,double,double> iter_parsll_edllfirst_edllfinal_llfinal;
	vector<int> vertex_indices_to_visit;
	
	for (int v_ind = n; v_ind < num_vertices; v_ind++) {
		vertex_indices_to_visit.push_back(v_ind);
	}
	unsigned seed = chrono::system_clock::now().time_since_epoch().count();
    mt19937 rng(seed);

    // Shuffle the vector
	cout << "randomizing the order in which nodes will be visited" << endl;
    shuffle(vertex_indices_to_visit.begin(), vertex_indices_to_visit.end(), rng);
	int v_ind;
	for (int v_i = 0; v_i < vertex_indices_to_visit.size(); v_i++){
        v_ind = vertex_indices_to_visit[v_i];
		v = (*this->vertexMap)[v_ind];
		cout << "node " << v_i+1 << ":" << v->name << endl;		
		if(v->degree != 3){
            throw mt_error("Expect internal nodes to have degree three");
        }
		loglikelihoodscoresForEachRepetition.clear();
		for (int rep = 0; rep < num_repetitions; rep++) {					
			iter_parsll_edllfirst_edllfinal_llfinal = this->EM_started_with_SSH_parameters_rooted_at(v);
			iter = get<0>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_pars = get<1>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_edc_first = get<2>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_edc_final = get<3>(iter_parsll_edllfirst_edllfinal_llfinal);
			logLikelihood_final = get<4>(iter_parsll_edllfirst_edllfinal_llfinal);
			loglikelihood_node_rep_file << v->name << "\t"										
										<< rep +1 << "\t"
										<< iter << "\t"
										<< setprecision(ll_precision) << logLikelihood_pars << "\t"
										<< setprecision(ll_precision) << loglikelihood_edc_first << "\t"
										<< setprecision(ll_precision) << loglikelihood_edc_final << "\t"
										<< setprecision(ll_precision) << logLikelihood_final << endl;
			if (max_log_likelihood < logLikelihood_final) {
				// this->WriteProbabilities();
				max_log_likelihood = logLikelihood_final;
			}
		}
	}
	loglikelihood_node_rep_file.close();
	cout << "max log likelihood obtained using SSH parameters is " << setprecision(ll_precision) << max_log_likelihood << endl;
	(*this->logFile) << "max log likelihood obtained using SSH parameters is " << setprecision(ll_precision) << max_log_likelihood << endl;
	return max_log_likelihood;		
}


void SEM::RootTreeByFittingAGMMViaEM() {
//	cout << "10a" << endl;
	this->RootTreeAtAVertexPickedAtRandom();	
//	cout << "10b" << endl;
	this->ComputeMPEstimateOfAncestralSequences();	
//	cout << "10c" << endl;
	this->ComputeInitialEstimateOfModelParameters();
//	cout << "10d" << endl;
//	string cliqueTreeFileNamePrefix = "/home/pk/Projects/EMTRasedForests/data/trees/cliqueTree_test_numberOfLeaves_16_replicate_1";
//	string cliqueTreeFileName;
	this->ClearAncestralSequences();
	double logLikelihood_current;
	int iter = 0;
	int maxIter = 100;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;	
	logLikelihood_current = 0;
//	this->ComputeLogLikelihood();
	// cout << "Initial loglikelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// (*this->logFile) << "Initial loglikelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;	
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		// cout << "Iteration no. " << iter << endl;
		// (*this->logFile) << "Iteration no. " << iter << endl;
//		cliqueTreeFileName = cliqueTreeFileNamePrefix + "_iter_" +to_string(iter);
//		chowLiuTreeFileName = chowLiuTreeFileNamePrefix + "_iter_" +to_string(iter);
//		MLRootedTreeFileName = MLRootedTreeFileNamePrefix + "_iter_" +to_string(iter);						
//		1. Construct clique tree		
		// if (verbose) {
		// 	cout << "Construct clique tree" << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
			
		// }		
		this->ConstructCliqueTree();				
//		this->WriteCliqueTreeToFile(cliqueTreeFileName);
//		2. Compute expected counts
		// if (verbose) {
		// 	cout << "Compute expected counts" << endl;
		// }	
		this->ComputeExpectedCountsForRootSearch();
		this->ComputePosteriorProbabilitiesUsingExpectedCounts();
//		this->WriteUnrootedTreeAsEdgeList(chowLiuTreeFileName);
		
//		3. Optimize model parameters
		// if (verbose) {
		// 	cout << "Optimize model parameters given expected counts" << endl;
		// }	
		this->ComputeMLRootedTreeForRootSearchUnderGMM();
//		4. Repeat steps 1 through 3 till convergence	
		// if (verbose) {
		// cout << "Expected loglikelihood for iteration " << iter << " is " << this->logLikelihood << endl;
		// (*this->logFile) << "Expected loglikelihood for iteration " << iter << " is " << this->logLikelihood << endl;
		// }	
		if ((this->logLikelihood > logLikelihood_current and (abs(this->logLikelihood - logLikelihood_current) > this->logLikelihoodConvergenceThreshold)) or (iter < 2 and iter < maxIter)) {
		// if ((this->logLikelihood > logLikelihood_current and (abs(this->logLikelihood - logLikelihood_current) > this->logLikelihoodConvergenceThreshold)) or iter < 2) {
			logLikelihood_current = this->logLikelihood;
		} else {
			continueIterations = 0;
		}
	}
	this->ComputeLogLikelihood();
}

/*
1. Number of iterations until EM converges
2. Initial log-likelihood with parsimony based parameters
3. Expected-data log-likelihood after one EM iteration
4. Expected-data log-likelihood after last EM iteration
5. Final log-likelihood using EM based parameters
6. Final location of root selected by EM
*/

tuple <int,double,double,double,double> SEM::EM_root_search_with_parsimony_rooted_at(SEM_vertex *v) {
	//	cout << "10a" << endl;
	// iterate over each internal node	
	this->RootTreeAtVertex(v);
	// cout << "10b" << endl;
	this->ComputeMPEstimateOfAncestralSequences();	
	// cout << "10c" << endl;
	this->ComputeInitialEstimateOfModelParameters();
	this->ComputeLogLikelihood();
	// cout << "Initial value of log-likelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "10d" << endl;
	this->ClearAncestralSequences();
	double logLikelihood_pars;
	double logLikelihood_exp_data_previous;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	logLikelihood_pars = this->logLikelihood;
	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// cout << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	logLikelihood_exp_data_previous = -1 * pow(10,4);
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		// if (verbose) {
		// 	cout << "Iteration no. " << iter << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
		// }
		// 1. Construct clique tree		
		// if (verbose) {
		// 	cout << "Construct clique tree" << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
			
		// }		
		
		this->ConstructCliqueTree();			
		// 2. Compute expected counts
		this->ComputeExpectedCountsForRootSearch();

		this->ComputePosteriorProbabilitiesUsingExpectedCounts();

		
		// 3. Optimize model parameters
		// if (verbose) {
		// 	cout << "Optimize model parameters given expected counts" << endl;
		// }
		
		// this->ComputeMLEstimateOfGMMGivenExpectedDataCompletion();
		this->ComputeMLRootedTreeForRootSearchUnderGMM();
		// this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		
		if (iter == 1){			
			logLikelihood_exp_data_first = this->logLikelihood;
            logLikelihood_exp_data_previous = this->logLikelihood;
		} else if ((this->logLikelihood > logLikelihood_exp_data_previous + this->logLikelihoodConvergenceThreshold) and (iter < this->maxIter)) {
			logLikelihood_exp_data_previous = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_previous;
		}
	}
	this->ComputeLogLikelihood();
		
	// cout << "log-likelihood computed by marginalization using EM parameters after iterations " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "Root location selected by EM is " << this->root->name << endl;
	// cout << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using EM parameters after iterations " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// (*this->logFile) <<  "Root location selected by EM is " << this->root->name << endl;
	// (*this->logFile) << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	
	
	return tuple<int,double,double,double,double>(iter,logLikelihood_pars,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
}

/*
1. Number of iterations until EM converges
2. Initial log-likelihood with SSH based parameters
3. Expected-data log-likelihood after one EM iteration
4. Expected-data log-likelihood after last EM iteration
5. Final log-likelihood using EM based parameters
*/

tuple <int,double,double,double,double> SEM::EM_started_with_SSH_parameters_rooted_at(SEM_vertex *v) {
	//	cout << "10a" << endl;
	// iterate over each internal node	
	this->RootTreeAtVertex(v);
	// cout << "10b" << endl;	
	this->SetInitialEstimateOfModelParametersUsingSSH();
	
	this->ComputeLogLikelihood();
	// cout << "Initial value of log-likelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;	
	// cout << "10d" << endl;
	this->ClearAncestralSequences();
	double logLikelihood_ssh;
	double logLikelihood_exp_data_previous;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	logLikelihood_ssh = this->logLikelihood;
	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// cout << "log-likelihood computed by marginalization using SSH parameters is " << setprecision(ll_precision) << logLikelihood_ssh << endl;
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using SSH parameters is " << setprecision(ll_precision) << logLikelihood_ssh << endl;
	logLikelihood_exp_data_previous = -1 * pow(10,4);
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		// if (verbose) {
		// 	cout << "Iteration no. " << iter << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
		// }
		// 1. Construct clique tree		
		// if (verbose) {
		// 	cout << "Construct clique tree" << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
			
		// }		
		
		this->ConstructCliqueTree();			
		// 2. Compute expected counts
		this->ComputeExpectedCountsForRootSearch();

		this->ComputePosteriorProbabilitiesUsingExpectedCounts();

		
		// 3. Optimize model parameters
		// if (verbose) {
		// 	cout << "Optimize model parameters given expected counts" << endl;
		// }
		
		this->ComputeMLEstimateOfGMMGivenExpectedDataCompletion();
		
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		
		if (iter == 1){
			logLikelihood_exp_data_previous = this->logLikelihood;
			logLikelihood_exp_data_first = this->logLikelihood;
		} else if ((this->logLikelihood > logLikelihood_exp_data_previous + this->logLikelihoodConvergenceThreshold) and (iter < this->maxIter)) {
			logLikelihood_exp_data_previous = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_previous;
		}
	}
	this->ComputeLogLikelihood();
		
	// cout << "log-likelihood computed by marginalization after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// (*this->logFile) << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	
	
	return tuple<int,double,double,double,double>(iter,logLikelihood_ssh,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
}


/*
1. Number of iterations until EM converges
2. Initial log-likelihood with parsimony based parameters
3. Expected-data log-likelihood after one EM iteration
4. Expected-data log-likelihood after last EM iteration
5. Final log-likelihood using EM based parameters
*/

tuple <int,double,double,double,double> SEM::EM_started_with_dirichlet_rooted_at(SEM_vertex *v) {
	//	cout << "10a" << endl;
	// iterate over each internal node	
	this->RootTreeAtVertex(v);	
	// cout << "10c" << endl;
	this->SetInitialEstimateOfModelParametersUsingDirichlet();
	this->ComputeLogLikelihood();
	// cout << "Initial value of log-likelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "10d" << endl;
	this->ClearAncestralSequences();
	double logLikelihood_pars;
	double logLikelihood_exp_data_current;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	logLikelihood_pars = this->logLikelihood;
	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// cout << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	logLikelihood_exp_data_current = -1 * pow(10,4);
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		// if (verbose) {
		// 	cout << "Iteration no. " << iter << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
		// }
		// 1. Construct clique tree		
		// if (verbose) {
		// 	cout << "Construct clique tree" << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
			
		// }		
		
		this->ConstructCliqueTree();			
		// 2. Compute expected counts
		this->ComputeExpectedCountsForRootSearch();

		this->ComputePosteriorProbabilitiesUsingExpectedCounts();

		
		// 3. Optimize model parameters
		// if (verbose) {
		// 	cout << "Optimize model parameters given expected counts" << endl;
		// }
		
		this->ComputeMLEstimateOfGMMGivenExpectedDataCompletion();
		
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		
		if (iter == 1){
			logLikelihood_exp_data_first = this->logLikelihood;
			logLikelihood_exp_data_current = this->logLikelihood;			
		} else if ((this->logLikelihood > logLikelihood_exp_data_current + this->logLikelihoodConvergenceThreshold) and (iter < this->maxIter)) {
			logLikelihood_exp_data_current = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_current;
		}
	}
	this->ComputeLogLikelihood();
		
	// cout << "log-likelihood computed by marginalization using EM parameters " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using EM parameters " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// (*this->logFile) << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	
	return tuple<int,double,double,double,double>(iter,logLikelihood_pars,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
}


/*
1. Number of iterations until EM converges
2. Initial log-likelihood with parsimony based parameters
3. Expected-data log-likelihood after one EM iteration
4. Expected-data log-likelihood after last EM iteration
5. Final log-likelihood using EM based parameters
*/

tuple <int,double,double,double,double> SEM::EM_started_with_parsimony_rooted_at(SEM_vertex *v) {
	//	cout << "10a" << endl;
	// iterate over each internal node	
	this->RootTreeAtVertex(v);
	// cout << "10b" << endl;
	this->ComputeMPEstimateOfAncestralSequences();	
	// cout << "10c" << endl;
	this->ComputeInitialEstimateOfModelParameters();
	this->ComputeLogLikelihood();
	// cout << "Initial value of log-likelihood is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "10d" << endl;
	this->ClearAncestralSequences();
	double logLikelihood_pars;
	double logLikelihood_exp_data_current;
	double logLikelihood_exp_data_first;
	double logLikelihood_exp_data_final;
	int iter = 0;	
	bool continueIterations = 1;
	this->debug = 0;
	bool verbose = 0;
	logLikelihood_pars = this->logLikelihood;
	// cout << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// cout << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	// (*this->logFile) << "-    -     -     -     -     -     -     -     -     -     -     -     -     -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using parsimony parameters is " << setprecision(ll_precision) << logLikelihood_pars << endl;
	logLikelihood_exp_data_current = -1 * pow(10,4);
	while (continueIterations) {
		// t_start_time = chrono::high_resolution_clock::now();
		iter += 1;
		// if (verbose) {
		// 	cout << "Iteration no. " << iter << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
		// }
		// 1. Construct clique tree		
		// if (verbose) {
		// 	cout << "Construct clique tree" << endl;
		// 	(*this->logFile) << "Iteration no. " << iter << endl;
			
		// }		
		
		this->ConstructCliqueTree();			
		// 2. Compute expected counts
		this->ComputeExpectedCountsForRootSearch();

		this->ComputePosteriorProbabilitiesUsingExpectedCounts();

		
		// 3. Optimize model parameters
		// if (verbose) {
		// 	cout << "Optimize model parameters given expected counts" << endl;
		// }
		
		this->ComputeMLEstimateOfGMMGivenExpectedDataCompletion();
		
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
		
		// cout << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		// (*this->logFile) << "log-likelihood computed using expected counts after EM iteration " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
		
		if (iter == 1){
			logLikelihood_exp_data_first = this->logLikelihood;
			logLikelihood_exp_data_current = this->logLikelihood;			
		} else if ((this->logLikelihood > logLikelihood_exp_data_current + this->logLikelihoodConvergenceThreshold) and (iter < this->maxIter)) {
			logLikelihood_exp_data_current = this->logLikelihood;
		} else {
			continueIterations = 0;
			logLikelihood_exp_data_final = logLikelihood_exp_data_current;
		}
	}
	this->ComputeLogLikelihood();
		
	// cout << "log-likelihood computed by marginalization using EM parameters " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// cout << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	// (*this->logFile) << "log-likelihood computed by marginalization using EM parameters " << iter << " is " << setprecision(ll_precision) << this->logLikelihood << endl;
	// (*this->logFile) << "- + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + - + -" << endl;
	
	return tuple<int,double,double,double,double>(iter,logLikelihood_pars,logLikelihood_exp_data_first,logLikelihood_exp_data_final,this->logLikelihood);
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
	double edgeLogLikelihood;	
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
	double logLikelihood_max = 0;
	int numberOfVerticesVisited = 0;	
	for (SEM_vertex * v : verticesToVisit) {
		numberOfVerticesVisited += 1;
		this->RootTreeAtVertex(v);		
		this->ComputeMLEstimateOfGMMGivenExpectedDataCompletion();
		this->ComputeLogLikelihoodUsingExpectedDataCompletion();
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


void SEM::ComputeMLRootedTreeForFullStructureSearch() {
	this->StoreEdgeListForChowLiuTree();
	// For each vertex v of the Chow-Liu tree
	SEM_vertex * v;
	double logLikelihood_max = 0;
	int verticesTried = 0;
	bool debug = 0;
	bool useExpectedLogLikForSelectingRoot = 1;
	string nonCanonicalRootedTreeFileName = "/home/pk/Projects/EMTRasedForests/data/trees/nonCanonicalRootedTree_test_numberOfLeaves_16_replicate_1";
	string canonicalRootedTreeFileName = "/home/pk/Projects/EMTRasedForests/data/trees/canonicalRootedTree_test_numberOfLeaves_16_replicate_1";
	for (pair <int, SEM_vertex *> idPtrPair : * this->vertexMap) {
		verticesTried += 1;
		// 	Root tree at v
		v = idPtrPair.second;
		// if (debug) {
		// 	cout << "Rooting tree at vertex " << v->name << endl;
		// }
		this->RootTreeAtVertex(v);		
		// if (debug) {
		// 	this->WriteRootedTreeAsEdgeList(nonCanonicalRootedTreeFileName);
		// 	// cout << "Is v an observed variable?" << endl;
		// 	if (v->observed) {
		// 		// cout << "Yes" << endl;
		// 	} else {
		// 		cout << "No" << endl;
		// 	}
		// 	cout << "Root name is " << this->root->name << endl;
		// }
		// Compute MLE of model parameters
		// assuming that posterior probabilities
		// P(V) (for each vertex) and 
		// P(V1, V2) (for each vertex pair) are available
//		cout << "Computing MLE of model parameters" << endl;
		this->ComputeMLEstimateOfGMMGivenExpectedDataCompletion();
		// Transform to bifurcating rooted tree
//		cout << "Transforming to bifurcating leaf-labeled tree" << endl;		
		if (useExpectedLogLikForSelectingRoot) {
			this->ComputeLogLikelihoodUsingExpectedDataCompletion();
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
		//	cout << "Current max loglikelihood is " << logLikelihood_max << endl;
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

Md SEM::GetP_yGivenx(Md P_xy) {
	Md P_yGivenx = Md{};
	array <double, 4> P_x;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		P_x[dna_x] = 0;
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			P_x[dna_x] += P_xy[dna_x][dna_y];
		}
	}
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			P_yGivenx[dna_x][dna_y] = P_xy[dna_x][dna_y] / P_x[dna_x];
		}
	}
	return (P_yGivenx);
}


void SEM::SetMinLengthOfEdges() {	
	for (pair<pair<SEM_vertex * , SEM_vertex * >,double> edgeAndLengthsPair: this->edgeLengths){		
		if (edgeAndLengthsPair.second < pow(10,-7)){
			this->edgeLengths[edgeAndLengthsPair.first]  = pow(10,-7);
		}
	}
}

 
void SEM::ComputeMLEstimateOfGMMGivenExpectedDataCompletion() {
	SEM_vertex * x; SEM_vertex * y;
	Md P_xy;
	for (pair <int, SEM_vertex*> idPtrPair : *this->vertexMap) {
		y = idPtrPair.second;
		x = y->parent;
		if (x != y) {
			if (x->id < y->id) {
				P_xy = this->posteriorProbabilityForVertexPair[make_pair(x,y)];
			} else {
				// Check following step				
				P_xy = MT(this->posteriorProbabilityForVertexPair[make_pair(y,x)]);
				
			}
			// MLE of transition matrices
			y->transitionMatrix = this->GetP_yGivenx(P_xy);
		} else {
			// MLE of root probability
			this->rootProbability = this->posteriorProbabilityForVertex[y];
			y->rootProbability = this->rootProbability;
			y->transitionMatrix = Md{};
			for (int i = 0; i < 4; i ++) {
				y->transitionMatrix[i][i] = 1.0;
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
	Md P;
	array <double, 4> rootProb_orig;
	array <double, 4> rootProb_new;
	vector <SEM_vertex *> childrenToSwap;
	bool checkForCanonicalForm;
	checkForCanonicalForm = IsTreeInCanonicalForm();
	string nonCanonicalRootedTreeFileName = "/home/pk/Projects/EMTRasedForests/data/trees/debugNonCanonicalRootedTree_before";
	this->WriteRootedTreeAsEdgeList(nonCanonicalRootedTreeFileName);
	int numberOfTransformations = 0;
	while (!checkForCanonicalForm) {
		numberOfTransformations += 1;
		if (numberOfTransformations > 10000) {
			cout << "Check Transformation of rooted tree to canonical form" << endl;
            throw mt_error("Check Transformation of rooted tree to canonical form");            
		}
		// Case 1. x->h
		// Check for hidden vertices that are leaves
		// Remove the arc x->h. This creates a singleton vertex h.
		tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeZeroAndInDegreeOne();		
		while (containsMatchingVertex) {
			// if (debug and containsMatchingVertex) {
			// 	cout << "Case 1. There is a hidden vertex that is a leaf" << endl;
			// }
			this->RemoveArc(h->parent,h);
			tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeZeroAndInDegreeOne();
		}
		// Case 2. p->h->c
		// Check for hidden vertices h with out degree 1 and in degree 1 
		// Remove p->h and h->c and add p->c
		tie (containsMatchingVertex, h) = this->CheckAndRetrieveHiddenVertexWithOutDegreeOneAndInDegreeOne();
		while (containsMatchingVertex) {
			// if (debug and containsMatchingVertex) {
			// 	cout << "Case 2. There is a non-root hidden vertex that has out degree 1" << endl;
			// }			
			p = h->parent;
			if (h->children.size() != 1){
                throw mt_error("Check case 2");
            }
			c = h->children[0];
			c->transitionMatrix = MM(h->transitionMatrix,c->transitionMatrix);
			h->transitionMatrix = this->I4by4;
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
			// if (debug) {
			// 	cout << "Case 3. The root is a hidden vertex with out degree 1" << endl;
			// }
			rootProb_orig = this->rootProbability;
			if (this->root->children.size()!=1){
                throw mt_error("Check case 3");
            }
			p = this->root;
			c = this->root->children[0];
			P = c->transitionMatrix;
			for (int y = 0; y < 4; y ++) {				
				rootProb_new[y] = 0;
				for (int x = 0; x < 4; x ++){
					rootProb_new[y] += rootProb_orig[x] * P[x][y];
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
			// if (debug) {
			// 	cout << "Case 4. The root is an observed vertex" << endl;
			// }
			tie (containsSingletonHiddenVertex, h_s) = this->CheckAndRetrieveSingletonHiddenVertex();
			if(!containsSingletonHiddenVertex){
                throw mt_error("Check case 4");
            }
            
			if(h_s->children.size() != 0) {
                throw mt_error("Check case 4");
            }
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
			// if (debug) {
			// 	cout << "Case 5. There is a non-leaf non-root observed vertex" << endl;
			// }
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
			// if (debug) {
			// 	cout << "Case 6. There is a multifurcation" << endl;
			// }
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

array <double, 4> SEM::GetBaseComposition(SEM_vertex * v) {
	array <double, 4> baseCompositionArray;
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

array <double, 4> SEM::GetObservedCountsForVariable(SEM_vertex * v) {
	array <double, 4> observedCounts;
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
			// if (debug) {
			// 	cout << "Estimated transition matrix is" << endl;
			// 	cout << c->transitionMatrix << endl;
			// }
		}
	}
	
}

void SEM::SetInitialEstimateOfModelParametersUsingDirichlet() {

	array <double, 4> pi_diri = sample_pi();

	this->rootProbability = pi_diri;
	this->root->rootProbability = pi_diri;

	SEM_vertex * p; SEM_vertex * c; Md M_pc;
	for (pair<SEM_vertex*,SEM_vertex*> edge : this->edgesForPostOrderTreeTraversal) {		
		p = edge.first;
		c = edge.second;

		for (int row = 0; row < 4; row++) {
			array <double, 4> M_row_diri = sample_M_row();
			int diri_index = 1;
			for (int col = 0; col < 4; col++) {
				if (row == col){
					M_pc[row][col] = M_row_diri[0];
				} else {
					M_pc[row][col] = M_row_diri[diri_index++];
				}
			}
		}		
		c->transitionMatrix = M_pc;
	}
}

void SEM::SetInitialEstimateOfModelParametersUsingSSH() {
	
	// cout << "root is set at " << this->root->name << endl;
	// set root probability 
	for (int x = 0; x < 4; x++) {
		this->rootProbability[x] = this->root->root_prob_hss[x];
		this->root->rootProbability[x] = this->root->root_prob_hss[x];
		// cout << "root probability for " << x << " is " << this->rootProbability[x] << endl;
	}


	SEM_vertex * p; SEM_vertex * c; Md M_pc;
	for (pair<SEM_vertex*,SEM_vertex*> edge : this->edgesForPostOrderTreeTraversal) {		
		p = edge.first;
		c = edge.second;		
		M_pc = (*this->M_hss)[{p,c}];	
		c->transitionMatrix = M_pc;
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
			// if (debug) {
			// 	cout << "Transition matrix for " << p->name << " to " << c->name << " is " << endl;
			// 	cout << c->transitionMatrix << endl;
			// }			
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
			if (c->compressedSequence[site] >= 4) {
                throw mt_error("Check dna in compressed sequence");
            }
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
	double currentProbability;
	map <SEM_vertex*, array<double,4>> conditionalLikelihoodMap;
	array <double,4> conditionalLikelihood;
	double maxProbability;
	double stateWithMaxProbability;
	double partialLikelihood;
	double siteLikelihood;
	double largestConditionalLikelihood = 0;
	double currentProb;
	unsigned char dna_ind_p; unsigned char dna_ind_c;
	vector <SEM_vertex *> verticesToVisit;
	SEM_vertex * p;
	SEM_vertex * c;
	Md P;
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
				conditionalLikelihoodMap.insert(pair<SEM_vertex *,array<double,4>>(c,conditionalLikelihood));
			}
			// Initialize conditional likelihood for ancestors
			if (conditionalLikelihoodMap.find(p) == conditionalLikelihoodMap.end()){
				for (unsigned char dna_c = 0; dna_c < 4; dna_c++){
				conditionalLikelihood[dna_c] = 1;
				}
				conditionalLikelihoodMap.insert(pair<SEM_vertex *,array<double,4>>(p,conditionalLikelihood));
			}
			largestConditionalLikelihood = 0;
			for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
				partialLikelihood = 0;
				for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
//					if (P[dna_p][dna_c]*conditionalLikelihoodMap[c][dna_c] == 0 and P[dna_p][dna_c] > 0 and conditionalLikelihoodMap[c][dna_c] > 0) {
//						cout << "Numerical underflow in computing partial likelihood" << endl;
//						cout << "P(y|x) is " << P[dna_p][dna_c] << endl;
//						cout << "L(y) is " << conditionalLikelihoodMap[c][dna_c] << endl;								
//						cout << "2^-256 is " << 1.0/pow(2,256) << endl;
//					}
					partialLikelihood += P[dna_p][dna_c]*conditionalLikelihoodMap[c][dna_c];
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
                throw mt_error("Largest conditional likelihood value is zero");                
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
					currentProbability = P[dna_ind_p][dna_ind_c];
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
	map <SEM_vertex*,Md> transitionMatrices;	
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
	double rowSum;
	double currentProb;
	double maxProb;	
	bool continueEM = 1;
	vector <SEM_vertex *> verticesToVisit;	
	SEM_vertex * p;
	SEM_vertex * c;
	Md P;
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
					P = Md{};
					for (int site = 0; site < this->numberOfSitePatterns; site++) {
						dna_p = p->compressedSequence[site];
						dna_c = c->compressedSequence[site];
						P[dna_p][dna_c] += this->sitePatternWeights[site];
					}
					for (unsigned char dna_p = 0; dna_p < 4; dna_p++) {
						rowSum = 0;
						for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
							rowSum += P[dna_p][dna_c];
						}
						for (unsigned char dna_c = 0; dna_c < 4; dna_c++) {
							 P[dna_p][dna_c] /= rowSum;
						}
					}
					c->transitionMatrix = P;
//					transitionMatrices.insert(pair<SEM_vertex*,Md>(c,P));
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
							partialLikelihood += P[dna_p][dna_c]*conditionalLikelihoodMap[c][dna_c];
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
						throw mt_error("Largest conditional likelihood value is zero");                        
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
							currentProb = P[dna_p][dna_c]*conditionalLikelihoodMap[c][dna_c];
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
	Md P;
	SEM_vertex * u;
	SEM_vertex * v;
	double sum;
	unsigned char dna_u; unsigned char dna_v;	
	for (unsigned int u_id = 0; u_id < this->vertexMap->size()-1; u_id ++) {
		u = (*this->vertexMap)[u_id];		
		// Posterior probability for vertex u
		u->posteriorProbability = this->GetBaseComposition(u);
		// Posterior probabilies for vertex pair (u,v)
		for (unsigned int v_id = u_id + 1 ; v_id < this->vertexMap->size()-1; v_id ++) {			
			v = (*this->vertexMap)[v_id];
			P = Md{};
			for (int site = 0; site < this->numberOfSitePatterns; site++ ) {		
				dna_u = u->compressedSequence[site];
				dna_v = v->compressedSequence[site];
				P[dna_u][dna_v] += this->sitePatternWeights[site];
			}
			sum = 0;
			for (dna_u = 0; dna_u < 4; dna_u ++) {				
				for (dna_v = 0; dna_v < 4; dna_v ++) {
					sum += P[dna_u][dna_v];
				}				
			}
			for (dna_u = 0; dna_u < 4; dna_u ++) {				
				for (dna_v = 0; dna_v < 4; dna_v ++) {
					P[dna_u][dna_v] /= sum;
				}				
			}			
			this->posteriorProbabilityForVertexPair.insert(make_pair(make_pair(u,v),P));
		}
		
	}	
}

double SEM::GetExpectedMutualInformation(SEM_vertex * x, SEM_vertex* y) {
	pair <SEM_vertex *, SEM_vertex *> vertexPair;
	if (x->id < y->id) {
		vertexPair = pair <SEM_vertex *, SEM_vertex *>(x,y);
	} else {
		vertexPair = pair <SEM_vertex *, SEM_vertex *>(y,x);
	}
	
	Md P = this->posteriorProbabilityForVertexPair[vertexPair];
//	cout << "Joint probability for vertex pair " << x->name << "\t" << y->name << " is " << endl;
//	cout << P << endl;
	std::array <double, 4> P_x;
	std::array <double, 4> P_y;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		P_x[dna_x] = 0;
		P_y[dna_x] = 0;
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			P_x[dna_x] += P[dna_x][dna_y];
			P_y[dna_x] += P[dna_y][dna_x];
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
	double mutualInformation = 0;
//	double inc;
	for (int dna_x = 0; dna_x < 4; dna_x ++) {
		for (int dna_y = 0; dna_y < 4; dna_y ++) {
			if (P[dna_x][dna_y] > 0) {
//				inc = P[dna_x][dna_y] * log(P[dna_x][dna_y]/(P_x[dna_x] * P_y[dna_y]));
//				cout << "Incrementing mutual information by " << inc << endl; 
				mutualInformation += P[dna_x][dna_y] * log(P[dna_x][dna_y]/(P_x[dna_x] * P_y[dna_y]));
			}
		}
	}
//	cout << "P_XY is " << endl << P << endl;
//	cout << "mutual information is " << mutualInformation << endl;
	return (mutualInformation);
}

void SEM::ComputeChowLiuTree() {
	this->ClearAllEdges();
	int numberOfVertices = this->vertexMap->size();
	for (int i = 0; i < numberOfVertices; i++) {
		if ((*this->vertexMap).find(i) == (*this->vertexMap).end()){
            throw mt_error("i should be present in vertex map");
        }
	}
	const int numberOfEdges = numberOfVertices * (numberOfVertices-1)/2;	
	double maxMutualInformation = 0;
	double mutualInformation;
	double * negMutualInformation;
	negMutualInformation = new double [numberOfEdges];		
	SEM_vertex * u; SEM_vertex * v;	
	int edgeIndex = 0;
	for (int i=0; i<numberOfVertices; i++) {
		u = (*this->vertexMap)[i];
		for (int j=i+1; j<numberOfVertices; j++) {
			v = (*this->vertexMap)[j];
			mutualInformation = this->GetExpectedMutualInformation(u,v);
			negMutualInformation[edgeIndex] = -1 * mutualInformation;
			if (mutualInformation > maxMutualInformation) {
				maxMutualInformation = mutualInformation;
			}
			edgeIndex += 1;
		}
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

	vector<int> p(numberOfVertices); 

	prim_graph p_graph(numberOfVertices, edges, negMutualInformation, numberOfEdges);
	
	prim(p_graph, &p[0]);
	
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
		}
	}	
	delete[] edges;
	delete[] negMutualInformation;
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
	// v is a central vertex	
	// Root tree at a randomly selected neighbor u of v
	uniform_int_distribution <int> distribution(0,v->neighbors.size()-1);
	u_ind = distribution(generator);
	u = v->neighbors[u_ind];
	this->root->AddChild(u);
	this->root->AddChild(v);
	u->AddParent(this->root);
	v->AddParent(this->root);
	verticesToVisit.clear();
	verticesVisited.clear();
	verticesToVisit.push_back(u);
	verticesToVisit.push_back(v);
	verticesVisited.push_back(u);
	verticesVisited.push_back(v);
	numberOfVerticesToVisit = verticesToVisit.size() - 1;
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
	this->SetLeaves();	
	this->SetEdgesForPreOrderTraversal();
	this->SetVerticesForPreOrderTraversalWithoutLeaves();
	this->SetEdgesForPostOrderTraversal();
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

void SEM::SetEdgesFromTopologyFile(){
	SEM_vertex * u; SEM_vertex * v;
	vector <string> splitLine;
	string u_name; string v_name; double t;
	t = 0.0;
	int num_edges = 0;
	vector <unsigned char> emptySequence;
	ifstream inputFile(this->topologyFileName.c_str());
	for(string line; getline(inputFile, line );) {
		num_edges++;
		vector<string> splitLine = split_ws(line);
		u_name = splitLine[0];
		v_name = splitLine[1];
		if (this->ContainsVertex(u_name)) {
			u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		} else {
			u = new SEM_vertex(this->h_ind,emptySequence);
			u->name = u_name;
			u->id = this->h_ind;
			this->vertexMap->insert(pair<int,SEM_vertex*>(u->id,u));
			this->nameToIdMap.insert(make_pair(u->name,u->id));			
			this->h_ind += 1;
			if(!this->ContainsVertex(u_name)){
                throw mt_error("check why u is not in vertex map");
            }
		}

		if (this->ContainsVertex(v_name)) {
			v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		} else {			
			v = new SEM_vertex(this->h_ind,emptySequence);
			v->name = v_name;
			v->id = this->h_ind;
			this->vertexMap->insert(pair<int,SEM_vertex*>(v->id,v));
			this->nameToIdMap.insert(make_pair(v->name,v->id));
			this->h_ind += 1;
			if(!this->ContainsVertex(v_name)){
                throw mt_error("Check why v is not in vertex map");
            }
		}
		u->AddNeighbor(v);
		v->AddNeighbor(u);		
		if (u->id < v->id) {
			this->edgeLengths.insert(make_pair(make_pair(u,v),t));			
		} else {
			this->edgeLengths.insert(make_pair(make_pair(v,u),t));
		}		
	}
	inputFile.close();
	cout << "number of edges in topology file is " << num_edges << endl;
}

void SEM::AddWeightedEdges(vector < tuple <string,string,double> > weightedEdgesToAdd) {
	SEM_vertex * u; SEM_vertex * v;
	string u_name; string v_name; double t;
	vector <unsigned char> emptySequence;
	for (tuple <string, string, double> weightedEdge : weightedEdgesToAdd) {
		tie (u_name, v_name, t) = weightedEdge;		
		if (this->ContainsVertex(u_name)) {
			u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		} else {
			if (!this->ContainsVertex(v_name)) {
				cout << "Adding edge " << u_name << "\t" << v_name << endl;
			}
			if(!this->ContainsVertex(v_name)){
                throw mt_error("Check why v is not in vertex map");
            }
			u = new SEM_vertex(this->h_ind,emptySequence);
			u->name = u_name;
			u->id = this->h_ind;
			this->vertexMap->insert(pair<int,SEM_vertex*>(u->id,u));
			this->nameToIdMap.insert(make_pair(u->name,u->id));			
			this->h_ind += 1;
			if(!this->ContainsVertex(u_name)){
                throw mt_error("Check why u is not in vertex map");
            }
		}
		
		if (this->ContainsVertex(v_name)) {
			v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		} else {
			if (!this->ContainsVertex(u_name)) {
				cout << "Adding edge " << u_name << "\t" << v_name << endl;
			}
			if(!this->ContainsVertex(u_name)){
                throw mt_error("Check why u is not in vertex map");
            }
			v = new SEM_vertex(this->h_ind,emptySequence);
			v->name = v_name;
			v->id = this->h_ind;
			this->vertexMap->insert(pair<int,SEM_vertex*>(v->id,v));
			this->nameToIdMap.insert(make_pair(v->name,v->id));
			this->h_ind += 1;
			if(!this->ContainsVertex(v_name)){
                throw mt_error("Check why v is not in vertex map");
            }
		}
		u->AddNeighbor(v);
		v->AddNeighbor(u);		
		if (u->id < v->id) {
			this->edgeLengths.insert(make_pair(make_pair(u,v),t));			
		} else {
			this->edgeLengths.insert(make_pair(make_pair(v,u),t));
		}
	}
}

void SEM::AddEdgeLogLikelihoods(vector<tuple<string,string,double>> edgeLogLikelihoodsToAdd) {
	SEM_vertex * u; SEM_vertex * v; double edgeLogLikelihood;
	string u_name; string v_name;
	pair<SEM_vertex *, SEM_vertex *> vertexPair;
	for (tuple<string,string,double> edgeLogLikelihoodTuple : edgeLogLikelihoodsToAdd) {
		tie (u_name, v_name, edgeLogLikelihood) = edgeLogLikelihoodTuple;
		u = (*this->vertexMap)[this->nameToIdMap[u_name]];
		v = (*this->vertexMap)[this->nameToIdMap[v_name]];
		vertexPair = pair <SEM_vertex *, SEM_vertex *> (u,v);
		this->edgeLogLikelihoodsMap.insert(pair<pair <SEM_vertex *, SEM_vertex *>,double>(vertexPair, edgeLogLikelihood));
	}	
}

void SEM::AddVertexLogLikelihoods(map<string,double> vertexLogLikelihoodsMapToAdd) {
	string v_name; SEM_vertex * v; double vertexLogLikelihood;
	for (pair<string,double> vNameAndLogLik : vertexLogLikelihoodsMapToAdd) {
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

double SEM::ComputeDistance(int v_i, int v_j) {
	vector <unsigned char> seq_i; vector <unsigned char> seq_j;	
	seq_i = (*this->vertexMap)[v_i]->compressedSequence;
	seq_j = (*this->vertexMap)[v_j]->compressedSequence;
	double sequence_length = 0;
	for (int site = 0; site < numberOfSitePatterns; site++) {
		sequence_length += double(this->sitePatternWeights[site]);
	}
	if(sequence_length <= 0){
        throw mt_error("Check pattern detection");
    }
	// logDet_distance
	double distance;	
	if (flag_Hamming) {
		distance = 0;
		for (int site = 0; site < numberOfSitePatterns; site++) {
			if (seq_i[site] != seq_j[site]) {
			distance += double(this->sitePatternWeights[site])/sequence_length;
			}						
		}				
	} else {
        throw mt_error("Check distance flag");
    }
	return (distance);
}


void SEM::ComputeNJTree() {	    
    map<pair<int,int>, double> distanceMap;
    map<int,double> R;
    vector<int> vertexIndsForIterating;
    vector<unsigned char> emptySequence;

    unsigned int n0 = this->numberOfObservedVertices;
    for (unsigned int i = 0; i < n0; ++i) {
        R[int(i)] = 0.0f;
        vertexIndsForIterating.push_back(int(i));
        for (unsigned int j = i + 1; j < n0; ++j) {
            double d = this->ComputeDistance(int(i), int(j));
            if (this->verbose) {
                	cout << "Distance measure for "
                          << (*this->vertexMap)[int(i)]->name << "\t"
                          << (*this->vertexMap)[int(j)]->name << "\t"
                          << this->distance_measure_for_NJ << "\t" << d << std::endl;
            }
            distanceMap[ord_pair(int(i), int(j))] = d;
            R[int(i)] += d;
            R[int(j)] += d;
        }
    }

    // NJ initialization
    const unsigned int Ninit = n0;
    if (Ninit < 3) return;

    for (unsigned int i = 0; i < Ninit; ++i) {
        R[int(i)] /= double(max<int>(int(Ninit) - 2, 1));
    }

    int next_internal = int(Ninit);

    // --- Main loop ---
    while (R.size() > 3) {
        double neighborDist = numeric_limits<double>::infinity();
        int i_selected = -1, j_selected = -1;

        // FIX: iterate over current vector size, not a stale n
        for (size_t ii = 0; ii < vertexIndsForIterating.size(); ++ii) {
            for (size_t jj = ii + 1; jj < vertexIndsForIterating.size(); ++jj) {
                int i = vertexIndsForIterating[ii];
                int j = vertexIndsForIterating[jj];
                auto it = distanceMap.find(ord_pair(i,j));
                if (it == distanceMap.end()) continue;
                double q = it->second - R[i] - R[j];
                if (q < neighborDist) { neighborDist = q; i_selected = i; j_selected = j; }
            }
        }

        if (i_selected < 0 || j_selected < 0) {
            cerr << "NJ: no selectable pair found; aborting.\n";
            throw mt_error("NJ: no selectable pair found; aborting.\n");
        }

        // Remove the chosen leaves and add new internal node
        vertexIndsForIterating.erase(remove(vertexIndsForIterating.begin(),
                                                vertexIndsForIterating.end(), i_selected),
                                                vertexIndsForIterating.end());
        vertexIndsForIterating.erase(remove(vertexIndsForIterating.begin(),
                                                vertexIndsForIterating.end(), j_selected),
                                                vertexIndsForIterating.end());
        vertexIndsForIterating.push_back(next_internal);

        // Create internal vertex
        SEM_vertex* h_ptr = new SEM_vertex(next_internal, emptySequence);
        h_ptr->name = "h_" + to_string(this->h_ind++);
        this->vertexMap->insert(make_pair(next_internal, h_ptr));

        // Connect
        (*this->vertexMap)[i_selected]->AddNeighbor((*this->vertexMap)[next_internal]);
        (*this->vertexMap)[j_selected]->AddNeighbor((*this->vertexMap)[next_internal]);
        (*this->vertexMap)[next_internal]->AddNeighbor((*this->vertexMap)[i_selected]);
        (*this->vertexMap)[next_internal]->AddNeighbor((*this->vertexMap)[j_selected]);

        // Update R and distances
        R.erase(i_selected);
        R.erase(j_selected);
        R[next_internal] = 0.0f;

        // New active size after merging (one fewer)
        const int n_active = int(vertexIndsForIterating.size());

        for (int kk = 0; kk < n_active - 1; ++kk) { // all except the new node at the back
            int k = vertexIndsForIterating[kk];
            double dik = distanceMap[ord_pair(i_selected, k)];
            double djk = distanceMap[ord_pair(j_selected, k)];
            double dij = distanceMap[ord_pair(i_selected, j_selected)];
            double newDist = 0.5f * (dik + djk - dij);

            // Update R[k]; note: (n_active-1) leaves excluding the new one
            R[k] = double(R[k] * (n_active - 1) - dik - djk + newDist) / double(std::max(1, n_active - 2));

            // Clean stale entries and set new distances
            distanceMap.erase(ord_pair(k, i_selected));
            distanceMap.erase(ord_pair(k, j_selected));
            distanceMap[ord_pair(k, next_internal)] = newDist;

            R[next_internal] += newDist;
        }

        // Average R for new node
        R[next_internal] /= double(std::max(1, n_active - 2));

        // Remove i-j entry
        distanceMap.erase(ord_pair(i_selected, j_selected));

        ++next_internal;
    }

    // Final join
    {
        SEM_vertex* h_ptr = new SEM_vertex(next_internal, emptySequence);
        h_ptr->name = "h_" + to_string(this->h_ind++);
        this->vertexMap->insert(make_pair(next_internal, h_ptr));
        for (int v : vertexIndsForIterating) {
            (*this->vertexMap)[v]->AddNeighbor((*this->vertexMap)[next_internal]);
            (*this->vertexMap)[next_internal]->AddNeighbor((*this->vertexMap)[v]);
        }
    }
}

void SEM::ComputeNJTree_may_contain_uninitialized_values() {
	
	map <pair <int,int>, double> distanceMap;
	map <int,double> R;
	vector <int> vertexIndsForIterating;
	vector <unsigned char> emptySequence;
	unsigned int n = this->numberOfObservedVertices;			
	double distance;		
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
	double neighborDist;
	double neighborDist_current;
	double newDist;
	int i; int j; int k;
	int i_selected; int j_selected;
	int h = n;
	
	while (R.size() > 3) {
		neighborDist = numeric_limits<double>::infinity();
		i_selected = j_selected = -1;

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
                R[k] = double(R[k]*(n-1)-distanceMap[pair<int,int>(k,i_selected)]-distanceMap[pair<int,int>(k,j_selected)] + newDist)/double(n-2);
                distanceMap.erase(pair<int,int>(k,i_selected));
				distanceMap.erase(pair<int,int>(k,j_selected));               
			} else if (j_selected < k) {
				newDist = distanceMap[pair<int,int>(i_selected,k)] + distanceMap[pair<int,int>(j_selected,k)];
                newDist -= distanceMap[pair<int,int>(i_selected,j_selected)];
                newDist *= 0.5;
                R[k] = double(R[k]*(n-1)-distanceMap[pair<int,int>(i_selected,k)]-distanceMap[pair<int,int>(j_selected,k)] + newDist)/double(n-2);
                distanceMap.erase(pair<int,int>(i_selected,k));
                distanceMap.erase(pair<int,int>(j_selected,k));
			} else {
			    newDist = distanceMap[pair<int,int>(i_selected,k)] + distanceMap[pair<int,int>(k,j_selected)];
                newDist -= distanceMap[pair<int,int>(i_selected,j_selected)];
                newDist *= 0.5;
                R[k] = double(R[k]*(n-1)-distanceMap[pair<int,int>(i_selected,k)]-distanceMap[pair<int,int>(k,j_selected)] + newDist)/double(n-2);
                distanceMap.erase(pair<int,int>(i_selected,k));
                distanceMap.erase(pair<int,int>(k,j_selected));
			}            
			distanceMap[pair<int,int>(k,h)] = newDist;
            R[h] += newDist;
		}
        R[h] /= double(n-2);
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


///...///...///...///...///...///...///... mst backbone manager ///...///...///...///...///...///...///...///...///

class EMTR
{
private:
	default_random_engine generator;
	vector <string> sequenceNames;
	map <string,unsigned char> mapDNAtoInteger;		
	ofstream emt_logFile;
	int numberOfLargeEdgesThreshold;
	int numberOfHiddenVertices = 0;
	int edgeWeightThreshold;	
	chrono::system_clock::time_point start_time;
	chrono::system_clock::time_point current_time;
	chrono::system_clock::time_point t_start_time;
	chrono::system_clock::time_point t_end_time;
	chrono::system_clock::time_point m_start_time;
	chrono::system_clock::time_point m_end_time;
	chrono::duration<double> timeTakenToComputeEdgeAndVertexLogLikelihoods;
	chrono::duration<double> timeTakenToComputeGlobalUnrootedPhylogeneticTree;
	chrono::duration<double> timeTakenToComputeSubtree;
	chrono::duration<double> timeTakenToComputeSupertree;
	chrono::duration<double> timeTakenToRootViaEdgeLoglikelihoods;
	chrono::duration<double> timeTakenToRootViaRestrictedSEM;
	string fastaFileName;
	string phylipFileName;
	string topologyFileName;
	string prefix_for_output_files;
	string ancestralSequencesString;
	string loglikelihood_node_rep_file_name;	
	string probabilityFileName_pars;
	string probabilityFileName_diri;
	string probabilityFileName_pars_root;
	string probabilityFileName_diri_root;
	string probabilityFileName_best;
	double max_ll_pars;
	double max_ll_diri;
	string MSTFileName;
	string GMMparametersFileName;
	string distance_measure_for_NJ = "Hamming";
	bool apply_patch = false;
	bool grow_tree_incrementally = false;
	bool flag_topology = false;
    bool flag_set_gmm_parameters = false;
	int ComputeHammingDistance(string seq1, string seq2);
	int ComputeHammingDistance(vector<unsigned char> recodedSeq1, vector<unsigned char> recodedSeq2);
	int GetEdgeIndex (int vertexIndex1, int vertexIndex2, int numberOfVertices);
	MST * M;
	SEM * P;
	SEM * p;
	void WriteOutputFiles();
	bool debug;
	bool verbose;
	bool localPhyloOnly;	
	bool useChowLiu;
	bool modelSelection; 
	double max_log_lik_pars;
	double max_log_lik_diri;
	double max_log_lik_ssh;
    int max_iter;		
	string supertree_method;
	int numberOfVerticesInSubtree;
	string GetSequenceListToWriteToFile(map <string, vector <unsigned char>> compressedSeqMap, vector <vector <int> > sitePatternRepetitions);
	vector <string> must_have;
	vector <string> may_have;
    int num_repetitions;
	int max_EM_iter;
	double conv_thresh;
public:
	void SetDNAMap();
	void SetThresholds();
	void EMTRackboneWithOneExternalVertex();
	void EMTRackbone_k2020_preprint();
	void EMgivenInputTopology();
	void RootSuperTree();
	void start_EMt_with_MPars(int num_repetitions);
	void start_EMt_with_SSH_pars(int num_repetitions);
	void EMTRackboneWithRootSEMAndMultipleExternalVertices();
	void EMTRackboneOverlappingSets();
	void EMTRackboneOnlyLocalPhylo();
	void main(string init_criterion, bool root_search);
    void EMparsimony();
	void EMdirichlet();
	void SetprobFileforSSH();
    void EMssh();
	string EncodeAsDNA(vector<unsigned char> sequence);
	vector<unsigned char> DecompressSequence(vector<unsigned char>* compressedSequence, vector<vector<int>>* sitePatternRepeats);
	EMTR(string sequenceFileNameToSet, string input_format, string topologyFileNameToSet, string prefix_for_output_files, int num_repetitions, int max_iter, double conv_threshold) {	
		// start_time = chrono::high_resolution_clock::now();
		this->prefix_for_output_files = prefix_for_output_files;
		this->emt_logFile.open(this->prefix_for_output_files + ".emt_log");
		// this->probabilityFileName = this->prefix_for_output_files + ".prob";
        if (input_format == "phylip") {
            cout << "file format is phylip\n";
            this->phylipFileName = sequenceFileNameToSet;
        } else if (input_format == "fasta") {            
            throw mt_error("Fasta file parsing not implemented");
        } else {
            throw mt_error("Sequence file format not recognized");
        }
		this->topologyFileName = topologyFileNameToSet;
		this->supertree_method = supertree_method;
        this->num_repetitions = num_repetitions;        
		this->verbose = 0;
		this->distance_measure_for_NJ = "Hamming";
		this->flag_topology = 1;
		this->conv_thresh = conv_threshold;
		this->max_EM_iter = max_iter;		
		this->numberOfLargeEdgesThreshold = 100;		
		MSTFileName = prefix_for_output_files + ".initial_MST";
		this->SetDNAMap();
		this->ancestralSequencesString = "";
		// this->m_start_time = chrono::high_resolution_clock::now();
		this->M = new MST();
        if (input_format == "phylip") {
            cout << "Reading phylip file\n";
            this->M->ReadPhyx(this->phylipFileName);
        }		
		this->M->ComputeMST();
	
		if (false) {
			this->M->WriteToFile(MSTFileName);
		}		
		// this->current_time = chrono::high_resolution_clock::now();		
	    
		int numberOfInputSequences = (int) this->M->vertexMap->size();
		this->M->SetNumberOfLargeEdgesThreshold(this->numberOfLargeEdgesThreshold);
		this->P = new SEM(1,conv_threshold,max_iter,this->verbose);
		this->P->SetStream(this->emt_logFile);		
		this->P->numberOfObservedVertices = numberOfInputSequences;
		this->P->logLikelihoodConvergenceThreshold = conv_threshold;
		this->P->maxIter = max_iter;
		// this->m_start_time = chrono::high_resolution_clock::now();
		this->EMgivenInputTopology();
		this->P->SetVertexVector();
        this->P->SetPrefixForOutputFiles(this->prefix_for_output_files);
		cout << "number of repetitions is set to " << this->num_repetitions << endl;
			}
	~EMTR(){
		delete this->P;
		delete this->M;	
	}
};

void EMTR::main(string init_criterion, bool root_search){
	this->P->init_criterion = init_criterion;
	this->P->root_search = root_search;	
}

void EMTR::EMparsimony() {
    cout << "Starting EM with initial parameters set using parsimony" << endl;
	this->P->probabilityFileName_pars = this->prefix_for_output_files + ".pars_prob";
	this->probabilityFileName_pars = this->prefix_for_output_files + ".pars_prob";
    this->max_log_lik_pars = this->P->EM_rooted_at_each_internal_vertex_started_with_parsimony(this->num_repetitions);
}


void EMTR::EMdirichlet() {
	cout << "Starting EM with initial parameters sampled from Dirichlet distribution" << endl;
	this->P->probabilityFileName_diri = this->prefix_for_output_files + ".diri_prob";
	this->probabilityFileName_diri = this->prefix_for_output_files + ".diri_prob";
	this->max_log_lik_diri = this->P->EM_rooted_at_each_internal_vertex_started_with_dirichlet(this->num_repetitions);
}

void EMTR::SetprobFileforSSH() {
	if (this->max_log_lik_pars > this->max_log_lik_diri) {
		cout << "Initializing with Parsimony yielded higher log likelihood score" << endl;
		this->probabilityFileName_best = this->probabilityFileName_pars;
	} else {
		this->probabilityFileName_best = this->probabilityFileName_diri;
		cout << "Initializing with Dirichlet yielded higher log likelihood score" << endl;
	}
}

void EMTR::EMssh() {
	this->SetprobFileforSSH();
	cout << "Starting EM with initial parameters set using Bayes rule as described in SSH paper" << endl;
	this->P->probabilityFileName_best = this->probabilityFileName_best;
	cout << this->P->probabilityFileName_best << endl;
    this->P->SetGMMparameters();
	this->P->ReparameterizeGMM();    
    this->max_log_lik_ssh = this->P->EM_rooted_at_each_internal_vertex_started_with_SSH_par(this->num_repetitions);
}

void EMTR::SetDNAMap() {
	this->mapDNAtoInteger["A"] = 0;
	this->mapDNAtoInteger["C"] = 1;
	this->mapDNAtoInteger["G"] = 2;
	this->mapDNAtoInteger["T"] = 3;
}

void EMTR::EMTRackboneOnlyLocalPhylo() {
	vector <string> names;
	vector <vector <unsigned char> > sequences;
	vector <int> sitePatternWeights;
	vector <vector <int> > sitePatternRepetitions;	
	vector <int> idsOfVerticesToRemove;
	vector <int> idsOfVerticesToKeep;
	vector <int> idsOfExternalVertices;
	vector <int> idsOfVerticesForSEM;
	vector <tuple <int, string, vector <unsigned char>>> idAndNameAndSeqTupleForVerticesToAdd;	
	//----##############################################################---//
	//	1.	Initialize the global phylogenetic tree P as the empty graph   //
	//----##############################################################---//
	// cout << "Starting MST-backbone" << endl;
	// cout << "1.	Initialize the global phylogenetic tree P as the empty graph" << endl;
	int numberOfInputSequences = (int) this->M->vertexMap->size();		
	current_time = chrono::high_resolution_clock::now();
	timeTakenToComputeEdgeAndVertexLogLikelihoods = chrono::duration<double>(current_time-current_time);	
	
	int largestIdOfVertexInMST = numberOfInputSequences;
	
	bool computeLocalPhylogeneticTree = 1;
	bool numberOfNonSingletonComponentsIsGreaterThanZero = 0;	
	
	while (computeLocalPhylogeneticTree) {
		// cout << "Number of vertices in MST is " << this->M->vertexMap->size() << endl;
		// this->emt_logFile << "Number of vertices in MST is " << this->M->vertexMap->size() << endl;
		// cout << "Max vertex degree in MST is " << this->M->maxDegree << endl;
		//----####################################################################---//
		//	2.	Compute the size of the smallest subtree ts = (Vs,Es) of M s.t.		 //
		//		|Vs| > s. Check if |Vm\Vs| > s.								   		 //
		// 		If yes then go to step 3 else go to step 9					   		 //
		// 		Bootstrapped alignments may contain zero-weight edges		   		 //
		//      If so then replace |Vs| with |{non-zero weighted edges in Es}| 		 //
		//      Additionally replace |Vm\Vs| with |{non-zero weighted edges in Es}|  //
		//----####################################################################---//				
		computeLocalPhylogeneticTree = this->M->ShouldIComputeALocalPhylogeneticTree();
		// cout << "2. Checking if local phylogenetic tree should be computed" << endl;
		if (computeLocalPhylogeneticTree) {
			//----####################################################################---//
			//	3.	Extract vertices inducing subtree (Vs), and external vertices (Ve)	 //
			//----####################################################################---//				
			// cout << "3. Extract vertices inducing subtree (Vs), and external vertices (Ve)" << endl;
			this->M->SetIdsOfExternalVertices();
			idsOfExternalVertices = this->M->idsOfExternalVertices;
			idsOfVerticesForSEM = this->M->subtree_v_ptr->idsOfVerticesInSubtree;
			this->numberOfVerticesInSubtree = this->M->subtree_v_ptr->idsOfVerticesInSubtree.size();
			for (int id: idsOfExternalVertices) {
				idsOfVerticesForSEM.push_back(id);
			}
			tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);
			//----########################################################---//
			//	4.	Compute local phylogeny t over (Vs U Ve) via SEM      	 //
			//----########################################################---//
			// cout << "4.	Compute local phylogeny t over (Vs U Ve) via SEM" << endl;			
			this->p = new SEM(largestIdOfVertexInMST,this->conv_thresh,this->max_EM_iter,this->verbose);
			this->p->AddSequences(sequences);
			this->p->SetNumberOfVerticesInSubtree(this->numberOfVerticesInSubtree);
			this->p->SetNumberOfInputSequences(numberOfInputSequences);
			this->p->AddRootVertex();
			this->p->AddNames(names);
			this->p->AddGlobalIds(idsOfVerticesForSEM);
			this->p->AddSitePatternWeights(sitePatternWeights);
			this->p->AddSitePatternRepeats(sitePatternRepetitions);			
			this->p->OptimizeTopologyAndParametersOfGMM();			
			// timeTakenToComputeUnrootedPhylogeny += chrono::duration_cast<chrono::seconds>(t_end_time - t_start_time);
			//----##################################################################---//	
			//  5.	Check if # of non-singleton components of forest f in p that       //
			//		is induced by Vs is greater than zero.							   //
			//		i.e., Does local phylogeny contain vertices/edges of interest?	   //
			//----##################################################################---//
			this->p->SelectIndsOfVerticesOfInterestAndEdgesOfInterest();
			numberOfNonSingletonComponentsIsGreaterThanZero = this->p->IsNumberOfNonSingletonComponentsGreaterThanZero();			
			// cout << "5. Checking if there are any vertices of interest" << endl;
			if (!numberOfNonSingletonComponentsIsGreaterThanZero) {
				//----####################################################---//	
				//  6.	If no then double subtree size and go to step 2 	 //
				//		else reset subtree size and go to to step 7		     //
				//----####################################################---//		
				this->M->doubleSubtreeSizeThreshold();
				// cout << "6. Doubling subtree size" << endl;
			} else {
				this->M->ResetSubtreeSizeThreshold();		
				//----################################---//
				//  7.	Add vertices/edges in f to P     //
				//----################################---//
				// cout << "7. Adding vertices/edges in f to P" << endl;
				this->p->RenameHiddenVerticesInEdgesOfInterestAndSetIdsOfVerticesOfInterest();
				this->p->SetWeightedEdgesToAddToGlobalPhylogeneticTree();				
				//this->P->AddWeightedEdges(this->p->weightedEdgesToAddToGlobalPhylogeneticTree);
				this->p->SetAncestralSequencesString();
				this->ancestralSequencesString += this->p->ancestralSequencesString;				
				t_start_time = chrono::high_resolution_clock::now();
				this->p->SetEdgeAndVertexLogLikelihoods();				
				// this->P->AddVertexLogLikelihoods(this->p->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree);
				// this->P->AddEdgeLogLikelihoods(this->p->edgeLogLikelihoodsToAddToGlobalPhylogeneticTree);
				t_end_time = chrono::high_resolution_clock::now();
				timeTakenToComputeEdgeAndVertexLogLikelihoods += t_end_time - t_start_time;
				// Add vertex logLikelihoods
				// Add edge logLikelihoods
				largestIdOfVertexInMST = this->p->largestIdOfVertexInMST;
				//----##############################---//
				//  8.	Update M and go to step 1	   //
				//----##############################---//
				// cout << "8. Updating MST" << endl;
				this->p->SetInfoForVerticesToAddToMST();				
				this->M->UpdateMSTWithMultipleExternalVertices(p->idsOfVerticesToKeepInMST, p->idsOfVerticesToRemove, p->idAndNameAndSeqTuple, idsOfExternalVertices);								
				delete this->p;
			}			
			computeLocalPhylogeneticTree = this->M->ShouldIComputeALocalPhylogeneticTree();
		}		
		// cout << "CPU time used for computing local phylogeny is " << chrono::duration<double>(t_end_time-t_start_time).count() << " second(s)\n";
		// this->emt_logFile << "CPU time used for computing local phylogeny is " << chrono::duration<double>(t_end_time-t_start_time).count() << " second(s)\n";			
	}	
	//----########################################################---//
	//	9.	Compute phylogenetic tree p over vertices in M, and      //
	//		add vertices/edges in p to P							 //
	//----########################################################---//
	cout << "Computing phylogenetic tree over all vertices in MST" << endl;
	this->M->UpdateMaxDegree();
	cout << "Max vertex degree in MST is " << this->M->maxDegree << endl;
	idsOfVerticesForSEM.clear();
	for (pair <int, MST_vertex *> idPtrPair: * this->M->vertexMap) {
		idsOfVerticesForSEM.push_back(idPtrPair.first);
	}
	cout << "Number of vertices in MST is " << idsOfVerticesForSEM.size() << endl;
	cout << "Number of edges in MST is " << this->M->edgeWeightsMap.size() << endl;
	tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);
	this->numberOfVerticesInSubtree = sequences.size();
	this->p = new SEM(largestIdOfVertexInMST,this->conv_thresh,this->max_EM_iter,this->verbose);
	this->p->SetFlagForFinalIterationOfSEM();
	this->p->AddSequences(sequences);
	this->p->SetNumberOfVerticesInSubtree(this->numberOfVerticesInSubtree);
	this->p->SetNumberOfInputSequences(numberOfInputSequences);
	this->p->AddRootVertex();
	this->p->AddNames(names);
	this->p->AddGlobalIds(idsOfVerticesForSEM);
	this->p->AddSitePatternWeights(sitePatternWeights);
	this->p->AddSitePatternRepeats(sitePatternRepetitions);	
	this->p->OptimizeTopologyAndParametersOfGMM();			
	this->p->SelectIndsOfVerticesOfInterestAndEdgesOfInterest();
	this->p->RenameHiddenVerticesInEdgesOfInterestAndSetIdsOfVerticesOfInterest();
	this->p->SetWeightedEdgesToAddToGlobalPhylogeneticTree();
	this->p->SetAncestralSequencesString();
	this->ancestralSequencesString += this->p->ancestralSequencesString;
	this->p->WriteAncestralSequences();	
	t_start_time = chrono::high_resolution_clock::now();
	this->p->SetEdgeAndVertexLogLikelihoods();
	t_end_time = chrono::high_resolution_clock::now();
	timeTakenToComputeEdgeAndVertexLogLikelihoods += t_end_time - t_start_time;
	delete this->p;
	// this->emt_logFile << "CPU time used for computing local phylogeny is " << chrono::duration<double>(t_end_time-t_start_time).count() << " second(s)\n";
}


void EMTR::EMTRackbone_k2020_preprint() {
	vector <string> names;
	vector <vector <unsigned char> > sequences;
	vector <int> sitePatternWeights;
	vector <vector <int> > sitePatternRepetitions;	
	vector <int> idsOfVerticesToRemove;
	vector <int> idsOfVerticesToKeep;
	vector <int> idsOfExternalVertices;
	vector <int> idsOfVerticesForSEM;
	vector <tuple <int, string, vector <unsigned char>>> idAndNameAndSeqTupleForVerticesToAdd;	
	//----##############################################################---//
	//	1.	Initialize the global phylogenetic tree P as the empty graph   //
	//----##############################################################---//
	// cout << "Starting MST-backbone" << endl;
	// cout << "1.	Initialize the global phylogenetic tree P as the empty graph" << endl;
	int numberOfInputSequences = (int) this->M->vertexMap->size();		
	current_time = chrono::high_resolution_clock::now();
	// timeTakenToComputeEdgeAndVertexLogLikelihoods = chrono::duration_cast<chrono::seconds>(current_time-current_time);
	
	// Initialize supertree
	idsOfVerticesForSEM.clear();
	for (pair <int, MST_vertex *> vIdAndPtr : * this->M->vertexMap) {
		idsOfVerticesForSEM.push_back(vIdAndPtr.first);
	}
	tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);	
	this->P->sequenceFileName = this->fastaFileName;
	this->P->AddSequences(sequences);
	this->P->AddNames(names);
	this->P->AddSitePatternWeights(sitePatternWeights);
	this->P->SetNumberOfInputSequences(numberOfInputSequences);	
	this->P->numberOfObservedVertices = numberOfInputSequences;
	// add duplicated sequences here
	
	int largestIdOfVertexInMST = numberOfInputSequences;
	
	bool computeLocalPhylogeneticTree = 1;
	bool numberOfNonSingletonComponentsIsGreaterThanZero = 0;	
	
	while (computeLocalPhylogeneticTree) {
		// current_time = chrono::high_resolution_clock::now();
		// cout << "Number of vertices in MST is " << this->M->vertexMap->size() << endl;				
		// this->emt_logFile << "Number of vertices in MST is " << this->M->vertexMap->size() << endl;
		// cout << "Max vertex degree in MST is " << this->M->maxDegree << endl;
		//----####################################################################---//
		//	2.	Compute the size of the smallest subtree ts = (Vs,Es) of M s.t.		 //
		//		|Vs| > s. Check if |Vm\Vs| > s.								   		 //
		// 		If yes then go to step 3 else go to step 9					   		 //
		// 		Bootstrapped alignments may contain zero-weight edges		   		 //
		//      If so then replace |Vs| with |{non-zero weighted edges in Es}| 		 //
		//      Additionally replace |Vm\Vs| with |{non-zero weighted edges in Es}|  //
		//----####################################################################---//				
		computeLocalPhylogeneticTree = this->M->ShouldIComputeALocalPhylogeneticTree();
		// cout << "2. Checking if local phylogenetic tree should be computed" << endl;
		if (computeLocalPhylogeneticTree) {
			//----####################################################################---//
			//	3.	Extract vertices inducing subtree (Vs), and external vertices (Ve)	 //
			//----####################################################################---//				
			// cout << "3. Extract vertices inducing subtree (Vs), and external vertices (Ve)" << endl;
			this->M->SetIdsOfExternalVertices();
			idsOfExternalVertices = this->M->idsOfExternalVertices;
			idsOfVerticesForSEM = this->M->subtree_v_ptr->idsOfVerticesInSubtree;
			this->numberOfVerticesInSubtree = this->M->subtree_v_ptr->idsOfVerticesInSubtree.size();
			for (int id: idsOfExternalVertices) {
				idsOfVerticesForSEM.push_back(id);
			}
			tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);
			//----########################################################---//
			//	4.	Compute local phylogeny p over (Vs U Ve) via SEM      	 //
			//----########################################################---//
			// cout << "4.	Compute local phylogeny p over (Vs U Ve) via SEM" << endl;			
			this->p = new SEM(largestIdOfVertexInMST,this->conv_thresh,this->max_EM_iter,this->verbose);
			this->p->AddSequences(sequences);
			this->p->SetNumberOfVerticesInSubtree(this->numberOfVerticesInSubtree);
			this->p->SetNumberOfInputSequences(numberOfInputSequences);
			this->p->AddRootVertex();
			this->p->AddNames(names);
			this->p->AddGlobalIds(idsOfVerticesForSEM);
			this->p->AddSitePatternWeights(sitePatternWeights);
			this->p->AddSitePatternRepeats(sitePatternRepetitions);			
			t_start_time = chrono::high_resolution_clock::now();
			this->p->OptimizeTopologyAndParametersOfGMM();
			t_end_time = chrono::high_resolution_clock::now();
			timeTakenToComputeSubtree = t_end_time - t_start_time;
			// cout << "CPU time used for computing subtree with " << this->p->numberOfObservedVertices << " leaves is " << timeTakenToComputeSubtree.count() << " seconds\n";
			// this->emt_logFile << "CPU time used for computing subtree with " << this->p->numberOfObservedVertices << " leaves is " << timeTakenToComputeSubtree.count() << " seconds\n";
			//----##################################################################---//	
			//  5.	Check if # of non-singleton components of forest f in p that       //
			//		is induced by Vs is greater than zero.							   //
			//		i.e., Does local phylogeny contain vertices/edges of interest?	   //
			//----##################################################################---//
			this->p->SelectIndsOfVerticesOfInterestAndEdgesOfInterest();
			numberOfNonSingletonComponentsIsGreaterThanZero = this->p->IsNumberOfNonSingletonComponentsGreaterThanZero();			
			// cout << "5. Checking if there are any vertices of interest" << endl;
			if (!numberOfNonSingletonComponentsIsGreaterThanZero) {
				//----####################################################---//	
				//  6.	If no then double subtree size and go to step 2 	 //
				//		else reset subtree size and go to to step 7		     //
				//----####################################################---//		
				this->M->doubleSubtreeSizeThreshold();
				// cout << "6. Doubling subtree size" << endl;
			} else {
				this->M->ResetSubtreeSizeThreshold();		
				//----################################---//
				//  7.	Add vertices/edges in f to P     //
				//----################################---//
				// cout << "7. Adding vertices/edges in f to P" << endl;
				this->p->RenameHiddenVerticesInEdgesOfInterestAndSetIdsOfVerticesOfInterest();
				this->p->SetWeightedEdgesToAddToGlobalPhylogeneticTree();				
				this->P->AddWeightedEdges(this->p->weightedEdgesToAddToGlobalPhylogeneticTree);
				this->p->SetAncestralSequencesString();
				this->ancestralSequencesString += this->p->ancestralSequencesString;				
				largestIdOfVertexInMST = this->p->largestIdOfVertexInMST;
				//----##############################---//
				//  8.	Update M and go to step 1	   //
				//----##############################---//
				// cout << "8. Updating MST" << endl;
				this->p->SetInfoForVerticesToAddToMST();				
				this->M->UpdateMSTWithMultipleExternalVertices(p->idsOfVerticesToKeepInMST, p->idsOfVerticesToRemove, p->idAndNameAndSeqTuple, idsOfExternalVertices);								
				delete this->p;
			}			
			computeLocalPhylogeneticTree = this->M->ShouldIComputeALocalPhylogeneticTree();
		}		
	}	
	//----########################################################---//
	//	9.	Compute phylogenetic tree t over vertices in M, and      //
	//		add vertices/edges in t to T							 //
	//----########################################################---//
	cout << "Computing phylogenetic tree over all vertices in MST" << endl;
	idsOfVerticesForSEM.clear();
	for (pair <int, MST_vertex *> idPtrPair: * this->M->vertexMap) {
		idsOfVerticesForSEM.push_back(idPtrPair.first);
	}
	tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);
	this->numberOfVerticesInSubtree = sequences.size();
	this->p = new SEM(largestIdOfVertexInMST,this->conv_thresh,this->max_EM_iter,this->verbose);
	this->p->SetFlagForFinalIterationOfSEM();
	this->p->AddSequences(sequences);
	this->p->SetNumberOfVerticesInSubtree(this->numberOfVerticesInSubtree);
	this->p->SetNumberOfInputSequences(numberOfInputSequences);
	this->p->AddRootVertex();
	this->p->AddNames(names);
	this->p->AddGlobalIds(idsOfVerticesForSEM);
	this->p->AddSitePatternWeights(sitePatternWeights);
	this->p->AddSitePatternRepeats(sitePatternRepetitions);	
	t_start_time = chrono::high_resolution_clock::now();
	this->p->OptimizeTopologyAndParametersOfGMM();
	t_end_time = chrono::high_resolution_clock::now();	
	timeTakenToComputeSubtree = t_end_time - t_start_time;
	// cout << "CPU time used for computing subtree with " << this->p->numberOfObservedVertices << " leaves is " << timeTakenToComputeSubtree.count() << " seconds\n";
	// this->emt_logFile << "CPU time used for computing subtree with " << this->p->numberOfObservedVertices << " leaves is " << timeTakenToComputeSubtree.count() << " seconds\n";
	this->p->SelectIndsOfVerticesOfInterestAndEdgesOfInterest();
	this->p->RenameHiddenVerticesInEdgesOfInterestAndSetIdsOfVerticesOfInterest();
	this->p->SetWeightedEdgesToAddToGlobalPhylogeneticTree();
	this->p->SetAncestralSequencesString();
	this->ancestralSequencesString += this->p->ancestralSequencesString;
	this->p->WriteAncestralSequences();	
	this->P->AddWeightedEdges(this->p->weightedEdgesToAddToGlobalPhylogeneticTree);		
	delete this->p;
	// cout << "Adding duplicated sequences to tree" << endl;
	// this->emt_logFile << "Adding duplicated sequences to tree" << endl;
	this->P->AddDuplicatedSequencesToUnrootedTree(this->M);
	this->P->WriteUnrootedTreeAsEdgeList(this->prefix_for_output_files + ".unrooted_edgeList");
	this->P->RootTreeAtAVertexPickedAtRandom();
	this->P->WriteRootedTreeInNewickFormat(this->prefix_for_output_files + ".unrooted_newick");	
}

void EMTR::start_EMt_with_SSH_pars(int num_repetitions) {
	this->P->EM_rooted_at_each_internal_vertex_started_with_SSH_par(num_repetitions);	
}

void EMTR::start_EMt_with_MPars(int num_repetitions){
	this->P->EM_rooted_at_each_internal_vertex_started_with_parsimony(num_repetitions);	
}
void EMTR::EMTRackboneWithRootSEMAndMultipleExternalVertices() {
	vector <string> names;
	vector <vector <unsigned char> > sequences;
	vector <int> sitePatternWeights;
	vector <vector <int> > sitePatternRepetitions;	
	vector <int> idsOfVerticesToRemove;
	vector <int> idsOfVerticesToKeep;
	vector <int> idsOfExternalVertices;
	vector <int> idsOfVerticesForSEM;
	vector <tuple <int, string, vector <unsigned char>>> idAndNameAndSeqTupleForVerticesToAdd;	
	//----##############################################################---//
	//	1.	Initialize the global phylogenetic tree T as the empty graph   //
	//----##############################################################---//
	// cout << "Starting MST-backbone" << endl;
	// cout << "1.	Initialize the global phylogenetic tree T as the empty graph" << endl;
	int numberOfInputSequences = (int) this->M->vertexMap->size();
	this->P = new SEM(1,this->conv_thresh,this->max_EM_iter,this->verbose);
	// Initialize global phylogeny
	idsOfVerticesForSEM.clear();
	for (pair <int, MST_vertex *> vIdAndPtr : * this->M->vertexMap) {
		idsOfVerticesForSEM.push_back(vIdAndPtr.first);
	}
	tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);	
	this->P->sequenceFileName = this->fastaFileName;
	this->P->AddSequences(sequences);
	this->P->OpenAncestralSequencesFile();
	this->P->AddNames(names);
	this->P->AddSitePatternWeights(sitePatternWeights);
	this->P->SetNumberOfInputSequences(numberOfInputSequences);	
	this->P->numberOfObservedVertices = numberOfInputSequences;
	
	int largestIdOfVertexInMST = numberOfInputSequences;
	
	bool computeLocalPhylogeneticTree = 1;
	bool numberOfNonSingletonComponentsIsGreaterThanZero = 0;	
	
	while (computeLocalPhylogeneticTree) {
		// cout << "Number of vertices in MST is " << this->M->vertexMap->size() << endl;
		// this->emt_logFile << "Number of vertices in MST is " << this->M->vertexMap->size() << endl;
		//----####################################################################---//
		//	2.	Compute the size of the smallest subtree ts = (Vs,Es) of M s.t.		 //
		//		|Vs| > s. Check if |Vm\Vs| > s.								   		 //
		// 		If yes then go to step 3 else go to step 9					   		 //
		// 		Bootstrapped alignments may contain zero-weight edges		   		 //
		//      If so then replace |Vs| with |{non-zero weighted edges in Es}| 		 //
		//      Additionally replace |Vm\Vs| with |{non-zero weighted edges in Es}|  //
		//----####################################################################---//				
		computeLocalPhylogeneticTree = this->M->ShouldIComputeALocalPhylogeneticTree();
		// cout << "2. Checking if local phylogenetic tree should be computed" << endl;
		if (computeLocalPhylogeneticTree) {
			//----####################################################################---//
			//	3.	Extract vertices inducing subtree (Vs), and external vertices (Ve)	 //
			//----####################################################################---//				
			// cout << "3. Extract vertices inducing subtree (Vs), and external vertices (Ve)" << endl;
			this->M->SetIdsOfExternalVertices();
			idsOfExternalVertices = this->M->idsOfExternalVertices;
			idsOfVerticesForSEM = this->M->subtree_v_ptr->idsOfVerticesInSubtree;
			this->numberOfVerticesInSubtree = this->M->subtree_v_ptr->idsOfVerticesInSubtree.size();
			for (int id: idsOfExternalVertices) {
				idsOfVerticesForSEM.push_back(id);
			}
			tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);
			//----########################################################---//
			//	4.	Compute local phylogeny t over (Vs U Ve) via SEM      	 //
			//----########################################################---//
			// cout << "4.	Compute local phylogeny t over (Vs U Ve) via SEM" << endl;
			this->p = new SEM(largestIdOfVertexInMST,this->conv_thresh,this->max_EM_iter,this->verbose);						
			this->p->sequenceFileName = this->fastaFileName;
			this->p->AddSequences(sequences);			
			this->p->SetNumberOfVerticesInSubtree(this->numberOfVerticesInSubtree);			
			this->p->SetNumberOfInputSequences(numberOfInputSequences);			
			this->p->AddNames(names);
			this->p->numberOfObservedVertices = sequences.size();			
			this->p->AddGlobalIds(idsOfVerticesForSEM);			
			this->p->AddSitePatternWeights(sitePatternWeights);			
			this->p->AddSitePatternRepeats(sitePatternRepetitions);			
			this->p->ComputeNJTree();
			t_start_time = chrono::high_resolution_clock::now();
			this->p->RootTreeByFittingAGMMViaEM();
			t_end_time = chrono::high_resolution_clock::now();
			this->p->ComputeMAPEstimateOfAncestralSequencesUsingCliques();
			// this->t->OptimizeTopologyAndParametersOfGMM();		
			//----##################################################################---//	
			//  5.	Check if # of non-singleton components of forest f in t that       //
			//		is induced by Vs is greater than zero.							   //
			//		i.e., Does local phylogeny contain vertices/edges of interest?	   //
			//----##################################################################---//
			this->p->SelectIndsOfVerticesOfInterestAndEdgesOfInterest();
			numberOfNonSingletonComponentsIsGreaterThanZero = this->p->IsNumberOfNonSingletonComponentsGreaterThanZero();			
			// cout << "5. Checking if there are any vertices of interest" << endl;
			if (!numberOfNonSingletonComponentsIsGreaterThanZero) {
				//----####################################################---//	
				//  6.	If no then double subtree size and go to step 2 	 //
				//		else reset subtree size and go to to step 7		     //
				//----####################################################---//		
				this->M->doubleSubtreeSizeThreshold();
				// cout << "6. Doubling subtree size" << endl;
			} else {
				this->M->ResetSubtreeSizeThreshold();		
				//----################################---//
				//  7.	Add vertices/edges in f to T     //
				//----################################---//
				// cout << "7. Adding vertices/edges in f to T" << endl;
				this->p->RenameHiddenVerticesInEdgesOfInterestAndSetIdsOfVerticesOfInterest();
				this->p->SetWeightedEdgesToAddToGlobalPhylogeneticTree();
				this->p->SetAncestralSequencesString();
				this->p->WriteAncestralSequences();
				this->P->AddWeightedEdges(this->p->weightedEdgesToAddToGlobalPhylogeneticTree);
				this->p->SetEdgeAndVertexLogLikelihoods();
				this->P->AddVertexLogLikelihoods(this->p->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree);
				this->P->AddEdgeLogLikelihoods(this->p->edgeLogLikelihoodsToAddToGlobalPhylogeneticTree);
				largestIdOfVertexInMST = this->p->largestIdOfVertexInMST;
				//----##############################---//
				//  8.	Update M and go to step 1	   //
				//----##############################---//
				// cout << "8. Updating MST" << endl;
				this->p->SetInfoForVerticesToAddToMST();
				this->M->UpdateMSTWithMultipleExternalVertices(p->idsOfVerticesToKeepInMST, p->idsOfVerticesToRemove, p->idAndNameAndSeqTuple, idsOfExternalVertices);				
			}			
			computeLocalPhylogeneticTree = this->M->ShouldIComputeALocalPhylogeneticTree();
			// cout << "CPU time used for computing local phylogeny is " << chrono::duration<double>(t_end_time-t_start_time).count() << " second(s)\n";
			// this->emt_logFile << "CPU time used for computing local phylogeny is " << chrono::duration<double>(t_end_time-t_start_time).count() << " second(s)\n";			
		}
	}	
	//----########################################################---//
	//	9.	Compute phylogenetic tree t over vertices in M, and      //
	//		add vertices/edges in t to T							 //
	//----########################################################---//
	cout << "Computing phylogenetic tree over remaining vertices" << endl;	
	idsOfVerticesForSEM.clear();
	for (pair <int, MST_vertex *> idPtrPair: * this->M->vertexMap) {
		idsOfVerticesForSEM.push_back(idPtrPair.first);
	}
	// cout << "Number of vertices in MST is " << idsOfVerticesForSEM.size() << endl;
	// this->emt_logFile << "Number of vertices in MST is " << this->M->vertexMap->size() << endl;
	tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);
	this->numberOfVerticesInSubtree = sequences.size();
	this->p = new SEM(largestIdOfVertexInMST,this->conv_thresh,this->max_EM_iter,this->verbose);
	this->p->SetFlagForFinalIterationOfSEM();
	this->p->sequenceFileName = this->fastaFileName;
	this->p->AddSequences(sequences);
	this->p->SetNumberOfVerticesInSubtree(this->numberOfVerticesInSubtree);
	this->p->SetNumberOfInputSequences(numberOfInputSequences);
	this->p->numberOfObservedVertices = sequences.size();
	this->p->AddNames(names);
	this->p->AddGlobalIds(idsOfVerticesForSEM);
	this->p->AddSitePatternWeights(sitePatternWeights);
	this->p->AddSitePatternRepeats(sitePatternRepetitions);
	this->p->ComputeNJTree();
	t_start_time = chrono::high_resolution_clock::now();
	this->p->RootTreeByFittingAGMMViaEM();
	t_end_time = chrono::high_resolution_clock::now();
	this->p->ComputeMAPEstimateOfAncestralSequencesUsingCliques();
	this->p->SelectIndsOfVerticesOfInterestAndEdgesOfInterest();
	this->p->RenameHiddenVerticesInEdgesOfInterestAndSetIdsOfVerticesOfInterest();
	this->p->SetWeightedEdgesToAddToGlobalPhylogeneticTree();
	this->p->SetAncestralSequencesString();
	this->p->WriteAncestralSequences();
	this->P->AddWeightedEdges(this->p->weightedEdgesToAddToGlobalPhylogeneticTree);	
	this->p->SetEdgeAndVertexLogLikelihoods();
	this->P->AddVertexLogLikelihoods(this->p->vertexLogLikelihoodsMapToAddToGlobalPhylogeneticTree);
	this->P->AddEdgeLogLikelihoods(this->p->edgeLogLikelihoodsToAddToGlobalPhylogeneticTree);
	// cout << "CPU time used for computing local phylogeny is " << chrono::duration<double>(t_end_time-t_start_time).count() << " second(s)\n";
	// this->emt_logFile << "CPU time used for computing local phylogeny is " << chrono::duration<double>(t_end_time-t_start_time).count() << " second(s)\n";			
	
	if(this->P->vertexMap->size() != this->P->edgeLengths.size() + 1){
        throw mt_error("P is not a tree");
    }
	//----##############---//		
	//	10.	Root T via EM  //
	//----##############---//
	// current_time = chrono::high_resolution_clock::now();
	// cout << "CPU time used for computing unrooted topology is " << chrono::duration<double>(current_time-start_time).count() << " second(s)\n";
	// this->emt_logFile << "CPU time used for computing unrooted topology is " << chrono::duration<double>(current_time-start_time).count() << " second(s)\n";
	// cout << "Rooting T by maximizing expected log likelihood" << endl;
	this->P->RootTreeBySumOfExpectedLogLikelihoods();
}


void EMTR::EMTRackboneWithOneExternalVertex() {
	this->P = new SEM(1,this->conv_thresh,this->max_EM_iter,this->verbose);
	cout << "Starting MST-backbone" << endl;
	bool subtreeExtractionPossible = 1;		
	vector <string> names;
	vector <vector <unsigned char> > sequences;
	vector <int> sitePatternWeights;
	vector <vector <int> > sitePatternRepetitions;
	vector <int> idsOfVerticesForSEM;
	vector <int> idsOfVerticesToRemove;
	vector <int> idsOfVerticesToKeep;
	vector <int> idsOfExternalVertices;
	vector <tuple <int, string, vector <unsigned char>>> idAndNameAndSeqTupleForVerticesToAdd;

	int h_ind = 1;
	
	// Initialize global phylogeny
	idsOfVerticesForSEM.clear();
	for (pair <int, MST_vertex *> vIdAndPtr : * this->M->vertexMap) {
		idsOfVerticesForSEM.push_back(vIdAndPtr.first);
	}
	tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);	
	this->P->sequenceFileName = this->fastaFileName;
	this->P->AddSequences(sequences);
	this->P->AddNames(names);
	this->P->AddSitePatternWeights(sitePatternWeights);
	cout << "Number of leaves is " << this->P->vertexMap->size() << endl;
	
	int numberOfRemainingVertices;
	int numberOfVerticesInSubtree;
	vector <string> weightedEdges;	
	string u_name; string v_name;
	vector <unsigned char> sequenceToAdd;
	string nameOfSequenceToAdd;
	vector <unsigned char> seq_u; vector <unsigned char> seq_v;
	vector <unsigned char> compressed_seq_u; vector <unsigned char> compressed_seq_v;
	map <int, int> EMVertexIndToPhyloVertexIdMap;	
	int subtreeSizeThreshold = this->M->numberOfLargeEdgesThreshold;
	cout << "Subtree size threshold is " << subtreeSizeThreshold << endl;
	// Iterate to completion
	MST_vertex * v_mst;
	tie (subtreeExtractionPossible, v_mst) = this->M->GetPtrToVertexSubtendingSubtree();
	numberOfRemainingVertices = this->M->vertexMap->size() - v_mst->idsOfVerticesInSubtree.size();
	cout << "numberOfRemainingVertices is " << numberOfRemainingVertices << endl;
	if (v_mst->idOfExternalVertex == -1 or numberOfRemainingVertices < 3) {
		subtreeExtractionPossible = 0;
	}
	cout << "Sequence length is " << this->P->sequenceLength << endl;
	while (subtreeExtractionPossible) {
		cout << "Number of vertices in MST is " << this->M->vertexMap->size() << endl;
		this->p = new SEM(h_ind,this->conv_thresh,this->max_EM_iter,this->verbose);	
		// ids of vertices in subtree
		idsOfVerticesForSEM = v_mst->idsOfVerticesInSubtree;
		numberOfVerticesInSubtree = v_mst->idsOfVerticesInSubtree.size();
		// ids of external vertices
		idsOfVerticesForSEM.push_back(v_mst->idOfExternalVertex);
		tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);
		h_ind += sequences.size() -2;
		// Perform SEM
		this->p->SetNumberOfVerticesInSubtree(numberOfVerticesInSubtree);
		this->p->AddSequences(sequences);
		this->p->AddRootVertex();
		this->p->AddNames(names);
		this->p->AddSitePatternWeights(sitePatternWeights);		
		this->p->OptimizeTopologyAndParametersOfGMM();		
		// Get edges to add
		
		this->P->AddWeightedEdges(this->p->weightedEdgesToAddToGlobalPhylogeneticTree);						
		sequenceToAdd = DecompressSequence(&this->p->compressedSequenceToAddToMST, &sitePatternRepetitions);			
		// edgeListFile << this->t->weightedEdgeListString;
		// Update MST
		
		this->M->UpdateMSTWithOneExternalVertex(v_mst->idsOfVerticesInSubtree, this->p->nameOfSequenceToAddToMST, sequenceToAdd);
		tie (subtreeExtractionPossible, v_mst) = this->M->GetPtrToVertexSubtendingSubtree();
		numberOfRemainingVertices = this->M->vertexMap->size() - v_mst->idsOfVerticesInSubtree.size();
		if (v_mst->idOfExternalVertex == -1 or numberOfRemainingVertices < 3) {
			subtreeExtractionPossible = 0;
		}
		delete this->p;	
	}		
	cout << "Number of remaining vertices in MST is " << this->M->vertexMap->size() << endl;		
	idsOfVerticesForSEM.clear();
	for (pair <int, MST_vertex *> vIdAndPtr : * this->M->vertexMap) {
		idsOfVerticesForSEM.push_back(vIdAndPtr.first);
	}	
	this->p = new SEM(h_ind,this->conv_thresh,this->max_EM_iter,this->verbose);
	tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);
	// cout << "Number of distinct site patterns is " << sitePatternWeights.size() << endl;
	this->p->SetFlagForFinalIterationOfSEM();
	this->p->numberOfExternalVertices = 1;
	this->p->AddSequences(sequences);
	this->p->AddRootVertex();
	this->p->AddNames(names);
	this->p->AddSitePatternWeights(sitePatternWeights);
	t_start_time = chrono::high_resolution_clock::now();
	this->p->OptimizeTopologyAndParametersOfGMM();
	t_end_time = chrono::high_resolution_clock::now();
	this->P->AddWeightedEdges(this->p->weightedEdgesToAddToGlobalPhylogeneticTree);
	delete this->p;	
}

void EMTR::EMgivenInputTopology(){
	vector <string> names;
	vector <vector <unsigned char> > sequences;
	vector <int> sitePatternWeights;
	vector <vector <int> > sitePatternRepetitions;		
	vector <int> idsOfVerticesForSEM;
	
	int numberOfInputSequences = (int) this->M->vertexMap->size();		
	idsOfVerticesForSEM.clear();
	for (pair <int, MST_vertex *> vIdAndPtr : * this->M->vertexMap) {
		idsOfVerticesForSEM.push_back(vIdAndPtr.first);
	}
	tie (names, sequences, sitePatternWeights, sitePatternRepetitions) = this->M->GetCompressedSequencesSiteWeightsAndSiteRepeats(idsOfVerticesForSEM);		
	cout << "setting sequence file name, topology file name, site pattern weights, number of input sequences" << endl;
    cout << "number of site patterns is " << sitePatternWeights.size() << endl;	
	this->P->sequenceFileName = this->fastaFileName;
	this->P->topologyFileName = this->topologyFileName;
	this->P->AddSequences(sequences);
	this->P->AddNames(names);
	this->P->AddSitePatternWeights(sitePatternWeights);
	this->P->SetNumberOfInputSequences(numberOfInputSequences);	
	this->P->numberOfObservedVertices = numberOfInputSequences;
	cout << "setting edges from topology file" << endl;		
	// replace the following and setting edges from input topology	
	this->P->SetEdgesFromTopologyFile();
}

vector<unsigned char> EMTR::DecompressSequence(vector<unsigned char>* compressedSequence, vector<vector<int>>* sitePatternRepeats){
	int totalSequenceLength = 0;
	for (vector<int> sitePatternRepeat: *sitePatternRepeats){
		totalSequenceLength += int(sitePatternRepeat.size());
	}
	vector <unsigned char> decompressedSequence;
	for (int v_ind = 0; v_ind < totalSequenceLength; v_ind++){
		decompressedSequence.push_back(char(0));
	}
	unsigned char dnaToAdd;
	for (int sitePatternIndex = 0; sitePatternIndex < int(compressedSequence->size()); sitePatternIndex++){
		dnaToAdd = (*compressedSequence)[sitePatternIndex];
		for (int pos: (*sitePatternRepeats)[sitePatternIndex]){
			decompressedSequence[pos] = dnaToAdd;
		}
	}
	return (decompressedSequence);	
}

string EMTR::EncodeAsDNA(vector<unsigned char> sequence){
	string allDNA = "AGTC";
	string dnaSequence = "";
	for (unsigned char s : sequence){
		dnaSequence += allDNA[s];
	}
	return dnaSequence;
}

string EMTR::GetSequenceListToWriteToFile(map <string, vector <unsigned char>> compressedSeqMap, vector <vector <int> > sitePatternRepetitions) {	
	vector <unsigned char> decompressedSequence;
	string dnaSequence;
	string u_name;
	string listOfVertexNamesAndDNAsequencesToWriteToFile;	
	for (pair <string,vector<unsigned char>> nameSeqPair : compressedSeqMap) {		
		decompressedSequence = this->DecompressSequence(&(nameSeqPair.second),&sitePatternRepetitions);
		dnaSequence = this->EncodeAsDNA(decompressedSequence);
		listOfVertexNamesAndDNAsequencesToWriteToFile += nameSeqPair.first + "\t" + dnaSequence + "\n"; 		
	}	
	return (listOfVertexNamesAndDNAsequencesToWriteToFile);
}

int EMTR::GetEdgeIndex (int vertexIndex1, int vertexIndex2, int numberOfVertices){
	int edgeIndex;
	edgeIndex = numberOfVertices*(numberOfVertices-1)/2;
	edgeIndex -= (numberOfVertices-vertexIndex1)*(numberOfVertices-vertexIndex1-1)/2;
	edgeIndex += vertexIndex2 - vertexIndex1 - 1;
	return edgeIndex;
}

int EMTR::ComputeHammingDistance(string seq1, string seq2) {
	int hammingDistance = 0;
	for (unsigned int i=0;i<seq1.length();i++){
		if (seq1[i] != seq2[i]){
			hammingDistance+=1;
		}		
	}
	return (hammingDistance);
};

int EMTR::ComputeHammingDistance(vector<unsigned char> recodedSeq1, vector<unsigned char> recodedSeq2) {
	int hammingDistance = 0;
	double ungappedSequenceLength = 0;
	for (unsigned int i=0;i<recodedSeq1.size();i++) {
		if (recodedSeq1[i] != recodedSeq2[i]) {
			hammingDistance+=1;
		}		
	}	
	return (hammingDistance);
};

PYBIND11_MODULE(emtr, m) {
    m.doc() = "pybind11 bindings for EMTR";
    m.def("ord_pair", &ord_pair);
    py::class_<EMTR>(m, "EMTR")
        .def(py::init<
                 string,  // sequence_file
                 string,  // seq_file_format
                 string,  // topology_file
				 string,   // prefix_for_output_files
                 int,     // num_repetitions
                 int,     // max_iter
                 double   // conv_threshold	 
             >(),
             py::arg("sequence_file"),
             py::arg("seq_file_format"),
             py::arg("topology_file"),
			 py::arg("prefix_for_output"),
             py::arg("num_repetitions"),
             py::arg("max_iter"),
             py::arg("conv_threshold")
             )
        .def("EMparsimony", &EMTR::EMparsimony)      
		.def("EMdirichlet", &EMTR::EMdirichlet)
        .def("EMssh", &EMTR::EMssh);
    py::register_exception<mt_error>(m, "mt_error");
}