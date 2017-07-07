#Implementation for paper 4: NetSimile: A Scalable Approach to Size-Independent Network Similarity

#Project Team Members:
#1. Manjusha Trilochan Awasthi (mawasth)
#2. Kriti Shrivastava (kshriva)
#3. Rachit Thirani (rthiran)

import networkx as nx
import numpy as np
import os
import sys
import scipy
from scipy.stats import kurtosis, skew

def readGraph(graphName):
    #Input: name of the graph for which the files have to be read
    #Output: creates and returns a list of networkx graphs
    filePath = os.getcwd() + '\datasets\\'+graphName
    # Read all files in the directory and create a networkx graph for each file
    graphList = list()
    for file in os.listdir(filePath):
        file = filePath + '\\' + file
        f = open(file, 'r')
        #Skipping the first row containing number of nodes and edges
        next(f)
        g = nx.Graph()
        for line in f:
           line = line.split()
           g.add_edge(int(line[0]),int(line[1]))
        #Append the graph into the list
        graphList.append(g)
    return graphList

def neighborhood(G, node, n):
    #Input: a networkx graph, node in the mentioned graph, and the radius for the neighborhood
    #Output: list of nodes in the graph G which are at a distance n from the node
     path_lengths = nx.single_source_dijkstra_path_length(G, node)
     return [node for node, length in path_lengths.items() if length == n]


#Algorithm 1: NETSIMILE
def netSimile(graphList, doClustering):
    #Input: a list of graphs for which anomaly has to be detected and variable stating whether clustering has to be performed
    #Create node-feature matrices for all the graphs
    nodeFeatureMatrices = getFeatures(graphList)
    #generate “signature” vectors for each graph
    signatureVectorList = aggregator(nodeFeatureMatrices)
    #do comparison and return similarity/distance values for the given graphs
   # compare(signatureVectorList, doClustering)


#Algorithm 2: NETSIMILE’s GETFEATURES
def getFeatures(list_graph_list):
    #Input: list of graphs for which feature list has to be generated
    #Output: a list of node*feature matrix for all the graphs in the graph list
    features = []
    for G in list_graph_list:
        F = []
        for V in G.nodes():
            d_i = len(G.neighbors(V))
            c_i = nx.clustering(G, V)
            d_ni = float(len(neighborhood(G, V, 2))) / float(d_i)
            c_ni = 0
            for neighbor in G.neighbors(V):
                c_ni = c_ni + nx.clustering(G, neighbor)
            c_ni = float(c_ni) / float(d_i)
            ego_gph = nx.ego_graph(G, V)
            E_ego = len(ego_gph.edges())

            Estar_ego = 0
            e_list = set()
            for v in ego_gph:
                e_list = e_list.union(G.edges(v))
            e_list = e_list - set(ego_gph.edges())
            Estar_ego = len(list(e_list))

            N_ego = 0
            n_list = set()
            for v in ego_gph:
                n_list = n_list.union(G.neighbors(v))
            n_list = n_list - set(ego_gph.nodes())
            N_ego = len(list(n_list))
            F.append([d_i, c_i, d_ni, c_ni, E_ego, Estar_ego, N_ego])
        features.append(F)
    return features

#Algorithm 3: NETSIMILE’s AGGREGATOR
def aggregator(nodeFeatureMatrices):
    #Input: a list of node*feature matrix for all the graphs in the list
    #Output: a list of signature vectors for all the graphs in the list
    signatureVectorList = list()
    for nodeFeatureMatrix in nodeFeatureMatrices:
        signatureVector = list()
        for i in range(7):
            list1 = [item[0] for item in nodeFeatureMatrix]
            mean1 = np.mean(list1)
            median1 = np.median(list1)
            stdev = np.std(list1)
            skewness = skew(list1)
            kurtosis1 = kurtosis(list1)
            aggFeature = [median1,mean1,stdev,skewness,kurtosis1]
            signatureVector.append(aggFeature)
        signatureVectorList.append(signatureVector)
    print(signatureVectorList)
    return(signatureVectorList)

#Algorithm 4: NETSIMILE’s COMPARE



if __name__ == "__main__":
    #Read input file name and create file path accordingly
    graphName = sys.argv[1]
    #Read files and create a graph list
    graphList = readGraph(graphName)
    #Setting doClustering as false as clustering is out of scope of this project
    doClustering = False
    # Algoritm 1: NetSimile
    netSimile(graphList, doClustering)