import os
import pandas as pd
import yaml
from rich.console import Console
from rich.table import Table
import json

CONSOLE = Console()

def parseConfigYAML(path):
    
    with open(path,'r') as f:
        params = yaml.load(f,yaml.SafeLoader)
    
    table = Table(title='Training params')
    
    table.add_column("Param",justify="left",no_wrap=True)
    table.add_column("Value",justify="left",no_wrap=True)
    for key,value in params.items():
        
        table.add_row(key,str(value))
        
    CONSOLE.print(table)
    return params

def logResults2File(file,data):
    with open(file,'w') as f:
        json.dump(data,f,indent=4)

def create_directory(directory_path):
    # Check if the directory already exists
    if not os.path.exists(directory_path):
        try:
            # Create the directory
            os.makedirs(directory_path)
            print(f"Directory '{directory_path}' created successfully.")
        except OSError as e:
            print(f"Error creating directory '{directory_path}': {e}")
    else:
        print(f"Directory '{directory_path}' already exists.")

    return directory_path

def loadEstimatorResults(path):

    return pd.read_csv(path)
