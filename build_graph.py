import pandas as pd
import torch
import os
from torch_geometric.data import HeteroData

print("Looking for file at:", os.path.abspath("data/drugs.csv"))

def load_ids(file_path):
    df = pd.read_csv(file_path)
    return df, {id_: idx for idx, id_ in enumerate(df.iloc[:, 0])}

def load_edges(file_path, src_map, tgt_map):
    df = pd.read_csv(file_path)
    src = [src_map[id_] for id_ in df.iloc[:, 0]]
    tgt = [tgt_map[id_] for id_ in df.iloc[:, 1]]
    edge_index = torch.tensor([src, tgt], dtype=torch.long)
    return edge_index

# Load nodes
drugs_df, drug_map = load_ids("data/drugs.csv")
diseases_df, disease_map = load_ids("data/diseases.csv")
side_df, side_map = load_ids("data/side_effects.csv")
proteins_df, protein_map = load_ids("data/proteins.csv")

# Init hetero data
data = HeteroData()
feature_dim = 16

data['drug'].x = torch.randn(len(drugs_df), feature_dim)
data['disease'].x = torch.randn(len(diseases_df), feature_dim)
data['side_effect'].x = torch.randn(len(side_df), feature_dim)
data['protein'].x = torch.randn(len(proteins_df), feature_dim)

# Load edges
data['drug', 'treats', 'disease'].edge_index = load_edges("data/edges/drug_treats_disease.csv", drug_map, disease_map)
data['drug', 'causes', 'side_effect'].edge_index = load_edges("data/edges/drug_causes_sideeffect.csv", drug_map, side_map)
data['drug', 'targets', 'protein'].edge_index = load_edges("data/edges/drug_targets_protein.csv", drug_map, protein_map)
data['side_effect', 'leads_to', 'disease'].edge_index = load_edges("data/edges/sideeffect_leads_to_disease.csv", side_map, disease_map)

print(data)
