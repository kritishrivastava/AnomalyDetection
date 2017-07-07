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
    compare(signatureVectorList, doClustering)


#Algorithm 2: NETSIMILE’s GETFEATURES
def getFeatures(graphList):
    #Input: list of graphs for which feature list has to be generated
    #Output: a list of node*feature matrix for all the graphs in the graph list
    nodeFeatureMatrices = []
    #Compute the node*feature matrix for graph G
    for G in graphList:
        nodeFeatureMatrix = []
        #Calculated the values of the 7 features for all nodes in the graph G
        for node in G.nodes():
            #Feature 1: degree of the node
            d_i = G.degree(node)
            #Feature 2: clustering coefficient of the node
            c_i = nx.clustering(G, node)
            #Feature 3: average number of node's two-hop away neighbors
            d_ni = float(len(neighborhood(G, node, 2))) / float(d_i)
            #Feature 4: average clustering coefficient of neighbors of the node
            c_ni = 0
            for neighbor in G.neighbors(node):
                c_ni = c_ni + nx.clustering(G, neighbor)
            c_ni = float(c_ni) / float(d_i)
            #Feature 5: number of edges in node's egonet
            egonet = nx.ego_graph(G, node)
            E_ego = len(egonet.edges())
            #Feature 6: number of outgoing edges from node's egonet
            Estar_ego = 0
            e_list = set()
            for vertex in egonet:
                #Finding all edges of the nodes in the egonet
                e_list = e_list.union(G.edges(vertex))
            #Removing the edges which are the part of egonet itself to get the outing edges
            e_list = e_list - set(egonet.edges())
            Estar_ego = len(list(e_list))
            #Feature 7: number of neighbors of egonet
            N_ego = 0
            n_list = set()
            for vertex in egonet:
                #Finding all neighbors of the nodes in the egonet
                n_list = n_list.union(G.neighbors(vertex))
            # Removing the nodes which are the part of egonet itself to get the remaining neighbors
            n_list = n_list - set(egonet.nodes())
            N_ego = len(list(n_list))            
            nodeFeatureMatrix.append([d_i, c_i, d_ni, c_ni, E_ego, Estar_ego, N_ego])
        #Append the node*graph matrix for the graph to the list
        nodeFeatureMatrices.append(nodeFeatureMatrix)
    return nodeFeatureMatrices

#Algorithm 3: NETSIMILE’s AGGREGATOR
def aggregator(nodeFeatureMatrices):
    #Input: a list of node*feature matrix for all the graphs in the list
    #Output: a list of signature vectors for all the graphs in the list
    signatureVectorList = list()
    for nodeFeatureMatrix in nodeFeatureMatrices:
        signatureVector = list()
        #Calculate the aggregate values for all the 7 features
        for i in range(7):
            featureColumn = [item[0] for item in nodeFeatureMatrix]
            aggFeature = [np.median(featureColumn),np.mean(featureColumn),np.std(featureColumn),
                          skew(featureColumn),kurtosis(featureColumn, fisher=False)]
            #Append the aggregated values for this feature to the signature vector
            signatureVector.append(aggFeature)
        signatureVectorList.append(signatureVector)
    return(signatureVectorList)

#Algorithm 4: NETSIMILE’s COMPARE
def compare(signatureVectorList, doClustering):
    #eff
    a = 0

if __name__ == "__main__":
    #Read input file name and create file path accordingly
    graphName = sys.argv[1]
    #Read files and create a graph list
    graphList = readGraph(graphName)
    #Setting doClustering as false as clustering is out of scope of this project
    doClustering = False
    # Algoritm 1: NetSimile
    netSimile(graphList, doClustering)