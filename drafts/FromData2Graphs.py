#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 16:53:15 2024

Python script to retrieve data from a specified URL or locally stored, 
read the data into pandas DataFrames, creates a network graph using NetworkX, 
and then saves the graph data as a GraphML file.

@author: molly
"""

from collections import defaultdict
import shutil
import numpy as np
import requests
import pandas as pd
import networkx as nx
import urllib.request
import os
import csv 

#TO DO: add date

_DATACACHE = "~/polygraphs-cache/data"

def validate_origin(origin):
    """
    Validates the origin, ensuring it's either a URL or a local file.
    """
    if not isinstance(origin, str):
        raise ValueError("File origin must be a string")

    remote = urllib.parse.urlparse(origin).scheme is not None #check if is a URL o local file 

    if remote:
        try:
            # Use requests library to check if URL is reachable
            response = requests.head(origin)
            response.raise_for_status()
        except Exception as e:
            raise Exception(f"Error validating origin: {e}")
    elif not os.path.isfile(origin):
        raise Exception(f"Local file does not exist: {origin}")

    return remote

def fetch_file(origin, destination_folder):
    """
    Downloads remote origin to destination folder or copies local file to folder.
    """
    if not os.path.isdir(destination_folder):
        os.makedirs(destination_folder)  # Create the destination folder if it doesn't exist

    if urllib.parse.urlparse(origin).scheme:
        # If remote file, download to destination folder
        filename = os.path.basename(urllib.parse.urlparse(origin).path)
        destination_path = os.path.join(destination_folder, filename)
        response = requests.get(origin)
        with open(destination_path, 'wb') as f:
            f.write(response.content)
    else:
        # If local file, copy to destination folder
        filename = os.path.basename(origin)
        destination_path = os.path.join(destination_folder, filename)
        shutil.copyfile(origin, destination_path)
    return destination_path

def txt_to_csv(input_file, output_file):
    """
    Converts a text file with ';' delimiter to a CSV file.

    Args:
        input_file (str): Path to the input text file.
        output_file (str): Path to the output CSV file.

    Returns:
        None
    """
    with open(input_file, 'r') as txt_file, open(output_file, 'w', newline='') as csv_file:
        txt_reader = csv.reader(txt_file, delimiter=';')
        csv_writer = csv.writer(csv_file)

        for row in txt_reader:
            csv_writer.writerow(row)

    print(f"Conversion completed. CSV file saved at: {output_file}")


#can add here a function to convert the csv in a gz file 

def read_csv(file):
    '''reads csv and ispects it to get source and destination for edges '''
    df = pd.read_csv(file)
    # Print column headings to see what we have in there 
    print("Columns in the DataFrame:")
    print(df.columns)
        
    # Prompt user to input column names for source and destination
    source_column_name = input("Enter the name of the source column: ")
    destination_column_name = input("Enter the name of the destination column: ")
        
    # Assign source and destination columns from DataFrame
    source = df[source_column_name]
    destination = df[destination_column_name]
        
    return df, source, destination
    

def create_graph(file):
    '''create a graph based on the dataframe'''
    G = nx.Graph()
    G.add_edges_from(zip(source,destination))
    print("The graph has", len(G), "nodes and", len(G.edges()), "edges")
    return G

def save_graph(graph, format_choice, destination_folder):
    """
    Saves the graph in the specified format (GML, GraphML, or GEXF) to the destination folder.

    """
    filename = "graph." + format_choice.lower()

    if format_choice.lower() == "gml":
        nx.write_gml(graph, os.path.join(destination_folder, filename))
        print("Graph saved as GML format.")
    elif format_choice.lower() == "graphml":
        nx.write_graphml(graph, os.path.join(destination_folder, filename))
        print("Graph saved as GraphML format.")
    elif format_choice.lower() == "gexf":
        nx.write_gexf(graph, os.path.join(destination_folder, filename))
        print("Graph saved as GEXF format.")
    else:
        print("Invalid file format. Please choose either GML, GraphML, or GEXF.")

#Example usage:
base = "http://sixdegreesoffrancisbacon.com/data/SDFB_relationships_2024_05_01.csv"
source = 'person1_index'
dest = 'person2_index'

if __name__ == "__main__":
    origin = input("Enter URL, local file path, or private repository URL: ")
    destination_folder = os.path.expanduser(_DATACACHE)

    # Determine the type of origin (URL or local file)
    is_remote = validate_origin(origin)
    if is_remote:
        try:
            downloaded_file = fetch_file(origin, destination_folder)
            print("Downloaded file:", downloaded_file)
        except Exception as e:
            print("Error:", e)

    # Check if the downloaded file is already a CSV, otherwise convert it to CSV
    if downloaded_file.endswith('.csv'):
        print("Input file is already a CSV file.")
    else:
        output_file = downloaded_file.replace('.txt', '.csv')
        if not os.path.exists(output_file):
            txt_to_csv(downloaded_file, output_file)
        else:
            print("Output file already exists. Please choose a different output file name.")
            
   
    # Read the CSV file and assign source and destination columns
    try: 
        df, source, destination = read_csv(downloaded_file)
        if df is not None:
            print("Source column:", source)
            print("Destination column:", destination) 
    except Exception as e:
        print("Error:", e)
        
    # Create the graph
    try:
        graph = create_graph(df)
        print("Graph created successfully.")
    except Exception as e:
        print("Error:", e)
        
    # Save the graph in one of the selected format
# Prompt the user for the desired file format
    file_format = input("Enter the desired file format (GML, GraphML, or GEXF): ")
    try:
        save_graph(graph, file_format, os.path.expanduser(_DATACACHE))
    except Exception as e:
        print("Error:", e)



