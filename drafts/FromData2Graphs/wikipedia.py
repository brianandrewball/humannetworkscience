#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 18:26:56 2024

@author: mariafedericanorelli
"""

import pandas as pd
import sys
import csv
import os
import networkx as nx
import torch
import dgl
from fromData2Graphs import create_graph, save_graph,normalize_graph #y9ou have to normalize 


module_path = '/Users/mariafedericanorelli/Desktop/humannetworkscience/drafts/FromData2Graphs'
sys.path.append(os.path.abspath(module_path))


# # Function to convert txt to csv with comma as delimiter
# def txt_to_csv(txt_file, csv_file, delimiter=','):
#     try:
#         with open(txt_file, 'r') as txt_f, open(csv_file, 'w', newline='') as csv_f:
#             writer = csv.writer(csv_f)
#             for line in txt_f:
#                 # Assuming fields are separated by commas in the txt file.
#                 row = line.strip().split(delimiter)
#                 writer.writerow(row)
#     except Exception as e:
#         print(f"An error occurred: {e}")

# # Load the dataset
# txt_file_path = '/Users/mariafedericanorelli/Desktop/humannetworkscience/datasets/temporal_test.txt'
# temporal_test = pd.read_csv(txt_file_path)

# # Extract source, destination, and timestamps
# source = temporal_test['Head']
# destination = temporal_test[' tail']  # Ensure no leading/trailing spaces in column names
# timestamps = temporal_test[' ts']  # Keep timestamps as integers or floats

# # Create and normalize the graph using the source and destination columns
# graph = create_graph(source, destination)
# graph = normalize_graph(graph)

# # Retrieve the original node IDs (if stored as 'original_id')
# node_mapping = nx.get_node_attributes(graph, 'original_id')

# # Initialize an empty dictionary for edge attributes (storing lists of timestamps)
# timestamp_dict = {}

# # Iterate through the rows and append the timestamp to the corresponding edge
# for i in range(len(source)):
#     # Map the source and destination to their corresponding graph node IDs
#     graph_src = list(node_mapping.keys())[list(node_mapping.values()).index(source.iloc[i])]
#     graph_dst = list(node_mapping.keys())[list(node_mapping.values()).index(destination.iloc[i])]
    
#     edge = (graph_src, graph_dst)
#     if edge not in timestamp_dict:
#         timestamp_dict[edge] = []  # Initialize an empty list for this edge
#     timestamp_dict[edge].append(int(timestamps.iloc[i]))  # Append the timestamp to the edge as an integer

# # Set the 'timestamp' as an edge attribute in the NetworkX graph
# nx.set_edge_attributes(graph, timestamp_dict, 'timestamp')

# # Convert the NetworkX graph directly to a DGL graph, preserving edge attributes
# dgl_graph = dgl.from_networkx(graph, edge_attrs=['timestamp'])

# # # Print edge data to check if the 'timestamp' attribute is present in the DGL graph
# # print(dgl_graph.edata)

# # # Check if 'timestamp' attribute exists in the edges
# # if 'timestamp' in dgl_graph.edata:
# #     print("Timestamp attribute found!")
# # else:
# #     print("No timestamp attribute found.")
    
# # # Load the temporal edge list without specifying the dtype_backend
# # temporal_edges = pd.read_csv('/Users/mariafedericanorelli/Desktop/humannetworkscience/datasets/tkgl-smallpedia/tkgl-smallpedia_edgelist.csv')


# # print(temporal_edges.head())
# # # Remove the 'Q' prefix from both head and tail columns
# # temporal_edges['head'] = temporal_edges['head'].str.replace('Q', '').astype(int)
# # temporal_edges['tail'] = temporal_edges['tail'].str.replace('Q', '').astype(int)

# # print(temporal_edges.head())
# # temporal_edges.to_csv('/Users/mariafedericanorelli/Desktop/humannetworkscience/datasets/tkgl-smallpedia/tkgl-smallpedia_base.csv', index=False)


# # # Add the path to your module from converting data to graph
# # module_path = '/Users/mariafedericanorelli/Desktop/humannetworkscience/drafts/FromData2Graphs'
# # sys.path.append(os.path.abspath(module_path))


# source = temporal_edges['head']
# destination = temporal_edges['tail']
# graph = create_graph(source, destination)
# graph = normalize_graph(graph)

# #save as gml
# destination_folder = '/Users/mariafedericanorelli/Desktop/humannetworkscience/graphs'
# save_graph(graph, "GML", destination_folder, "smallpedia_temporal_graph_normalized")

# # Assign the 'timestamp' column as a node attribute
# timestamps = pd.Series(temporal_edges['ts'], index=temporal_edges['head'].to_dict())
# nx.set_node_attributes(graph, timestamps, 'ts')

#FOR STATIC DATA
# Load the static edge list
static_edges = pd.read_csv('/Users/mariafedericanorelli/Desktop/humannetworkscience/datasets/tkgl-smallpedia/tkgl-smallpedia_static_edgelist.csv')

print(static_edges.head())


# Remove the 'Q' prefix from both head and tail columns
static_edges['head'] = static_edges['head'].str.replace('Q', '').astype(int)
static_edges['tail'] = static_edges['tail'].str.replace('Q', '').astype(int)

print(static_edges.head())
static_edges.to_csv('/Users/mariafedericanorelli/Desktop/humannetworkscience/datasets/tkgl-smallpedia/tkgl-smallpedia_base.csv', index=False)

#CREATE GML FILE

# Add the path to your module from converting data to graph
module_path = '/Users/mariafedericanorelli/Desktop/humannetworkscience/drafts/FromData2Graphs'
sys.path.append(os.path.abspath(module_path))
from fromData2Graphs import create_graph, save_graph,normalize_graph #y9ou have to normalize 

source = static_edges['head']
destination = static_edges['tail']
graph = create_graph(source, destination)
graph = normalize_graph(graph)

#save as gml
destination_folder = '/Users/mariafedericanorelli/Desktop/humannetworkscience/graphs'
save_graph(graph, "GML", destination_folder, "smallpedia_static_graph_normalized")

# # Path to the GML file
# gml_file = os.path.expanduser('/Users/mariafedericanorelli/Desktop/humannetworkscience/graphs/smallpedia_static_graph_normalized.gml')

# # Load the graph from the GML file using NetworkX
# G = nx.read_gml(gml_file, destringizer=int)

# edges = []
# node_ids = set()
    
# for edge in list(nx.to_edgelist(G)):
#     # Extract source and destination nodes
#     node1, node2 = edge[0], edge[1]
        
#     # Check that both edge ids are integers
#     if not all(isinstance(x, int) for x in (node1, node2)):
#         raise ValueError("GML File: Node IDs should be specified as integers")
        
#     # Add nodes to the set
#     node_ids.add(node1)
#     node_ids.add(node2)
        
#     # Add the edge as a tensor
#     edges.append(torch.tensor((node1, node2)))
    
# # Create a DGL graph with explicit node IDs
# Graph = dgl.graph(edges, num_nodes=len(node_ids))


# # Print graph details to verify
# print(f"Graph created with {Graph.num_nodes()} nodes and {Graph.num_edges()} edges")