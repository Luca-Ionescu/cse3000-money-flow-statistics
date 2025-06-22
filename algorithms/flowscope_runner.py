import spartan as st
import pandas as pd
import os
import glob
import matplotlib.pyplot as plt
import numpy as np
from spartan.model.flowscope import flowscopeFraudDect


def run_flowscope(k:int):
    # file_list = [f"./output/fs{i}.csv" for i in range(1, k+1)]
    # df_combined = pd.concat((pd.read_csv(f, header=0, names=["idx", "src", "dst", "amount"]) for f in file_list), ignore_index = True)
    # unique_ids = pd.Index(df_combined["src"]).append(pd.Index(df_combined["dst"])).unique()
    #
    # id_to_int = {acc_id: i for i, acc_id in enumerate(unique_ids)}
    # int_to_id = {i: acc_id for acc_id, i in id_to_int.items()}
    #
    # for i in range(1, k+1):
    #     df = pd.read_csv("./output/fs" + str(i) + ".csv", index_col=0)
    #     df["0"] = df["0"].map(id_to_int)
    #     df["1"] = df["1"].map(id_to_int)
    #
    #     df.to_csv("./encoded-output/fs" + str(i) + ".csv", columns=['0', '1', '2'], index=False, header=False)

    offset = 0
    id_to_int_per_layer =[] # global mapping from int index -> account ID
    int_to_id_per_layer = []

    for i in range(1, k + 1):
        int_to_id = {}
        df = pd.read_csv(f"./output/fs{i}.csv", header=0, names=["idx", "src", "dst", "amount"])

        if i == 1:
            unique_ids_src = pd.Index(df["src"]).unique()
            id_to_int = {acc_id: idx + offset for idx, acc_id in enumerate(unique_ids_src)}
            offset += len(unique_ids_src)
            id_to_int_per_layer.append(id_to_int)
            for acc_id, idx in id_to_int.items():
                int_to_id[idx] = acc_id
            int_to_id_per_layer.append(int_to_id)

            unique_ids_dst = pd.Index(df["dst"]).unique()
            id_to_int = {acc_id: idx + offset for idx, acc_id in enumerate(unique_ids_dst)}
            offset += len(unique_ids_dst)
            id_to_int_per_layer.append(id_to_int)
            int_to_id = {}
            for acc_id, idx in id_to_int.items():
                int_to_id[idx] = acc_id
            int_to_id_per_layer.append(int_to_id)

        else:
            unique_ids = pd.Index(df["dst"]).unique()

            # Map for this file: account ID -> integer starting from offset
            id_to_int = {acc_id: idx + offset for idx, acc_id in enumerate(unique_ids)}

            # Update global int_to_id with new mappings
            for acc_id, idx in id_to_int.items():
                int_to_id[idx] = acc_id

            id_to_int_per_layer.append(id_to_int)
            int_to_id_per_layer.append(int_to_id)
            offset += len(unique_ids)


        df["src"] = df["src"].map(id_to_int_per_layer[i-1])
        df["dst"] = df["dst"].map(id_to_int_per_layer[i])

        # Save encoded file
        df.to_csv(f"./encoded-output/fs{i}.csv", columns=["src", "dst", "amount"], index=False, header=False)


    tensor_data = []

    for i in range(1, k+1):
        tensor_data.append(st.loadTensor(path= "./encoded-output/fs" + str(i) + ".csv", header=0))

    tensor_array = []
    for tens in tensor_data:
        stensor = tens.toSTensor(hasvalue=True)

        tensor_array.append(stensor)

    for tens_idx in range(len(tensor_array) -1):
        maxshape = max(tensor_array[tens_idx].shape[1], tensor_array[tens_idx + 1].shape[0])
        tensor_array[tens_idx].shape = (tensor_array[tens_idx].shape[0], maxshape)
        tensor_array[tens_idx + 1].shape = (maxshape, tensor_array[tens_idx + 1].shape[1])

    graphs = []
    for tens in tensor_array:
        print("STensor shape:", tens.shape)
        graphs.append(st.Graph(tens, bipartite=True, weighted=True, modet=None))

    fs = st.FlowScope(graphs)
    res = fs.run(k=3, alpha=3)
    print(res)
    return res, int_to_id_per_layer

def run_flowscope_no_encoding(k:int):

    tensor_data = []

    for i in range(1, k):
        tensor_data.append(st.loadTensor(path= "./encoded-output/fs" + str(i) + ".csv", header=0))

    tensor_array = []
    for tens in tensor_data:
        stensor = tens.toSTensor(hasvalue=True)
        print("STensor shape:", stensor.shape)
        tensor_array.append(stensor)

    graphs = []
    for tens in tensor_array:
        graphs.append(st.Graph(tens, bipartite=True, weighted=True, modet=None))

    ad_model = st.AnomalyDetection.create(graphs, st.ADPolicy.FlowScope, 'flowscope')
    res = ad_model.run(k=k, alpha=4)
    print(res)
    return res


def run_flowscope_spartan(input_dir, k=3):
    """
    Load bipartite CSVs from input_dir, run FlowScope via Spartan.

    Args:
        input_dir: Directory containing fs1.csv, fs2.csv, ..., fs{k-1}.csv
        k: number of layers
    """

    graph_list = []
    stensors = []

    # Load each bipartite CSV as Spartan tensor
    for i in range(1, k):
        csv_path = f"{input_dir}/fs{i}.csv"
        tensor_data = st.loadTensor(path=csv_path, header=None)
        stensor = tensor_data.toSTensor(hasvalue=True)
        stensors.append(stensor)

    # Adjust intermediate dimensions for compatibility
    for i in range(len(stensors) - 1):
        max_dim = max(stensors[i].shape[1], stensors[i + 1].shape[0])
        stensors[i].shape = (stensors[i].shape[0], max_dim)
        stensors[i + 1].shape = (max_dim, stensors[i + 1].shape[1])

    # Create Spartan Graph objects
    for stensor in stensors:
        graph = st.Graph(stensor, bipartite=True, weighted=True, modet=None)
        graph_list.append(graph)

    # Create and run FlowScope anomaly detection model
    ad_model = st.AnomalyDetection.create(graph_list, st.ADPolicy.FlowScope, 'flowscope')
    results = ad_model.run(k=k, alpha=4, maxsize=(-1,) * k)

    #print(results)

    return results


def run_flowscope_new(layered_tensors, k):
    """
    Run FlowScope on multipartite graphs with arbitrary k layers.

    Args:
        layered_tensors: list of pandas DataFrames, each representing edges between layers l and l+1.
        k: number of layers in the multipartite graph.

    Returns:
        The anomaly detection model result from FlowScope.
    """

    stensors = []
    # Convert each DataFrame to CSV and load as Spartan sparse tensor
    for i, df in enumerate(layered_tensors):
        temp_csv = f"temp_layer_{i+1}.csv"
        df.to_csv(temp_csv, index=False, header=False)
        stensor = st.loadTensor(path=temp_csv, header=None)
        stensors.append(stensor.toSTensor(hasvalue=True))

    # Align intermediate dimensions for all tensors
    # For FlowScope, consecutive tensors share a middle dimension:
    # shape of tensor l: (nodes in layer l, nodes in layer l+1)
    for i in range(len(stensors) - 1):
        maxshape = max(stensors[i].shape[1], stensors[i+1].shape[0])
        stensors[i].shape = (stensors[i].shape[0], maxshape)
        stensors[i+1].shape = (maxshape, stensors[i+1].shape[1])

    # Build graph objects for each tensor
    graphs = [st.Graph(t, bipartite=True, weighted=True, modet=None) for t in stensors]

    # Create the layered graph list expected by FlowScope
    step2list = graphs

    # Run FlowScope
    fs = st.FlowScope(step2list)
    ad_model = st.AnomalyDetection.create(step2list, st.ADPolicy.FlowScope, 'flowscope')

    results = ad_model.run(k=k, alpha=4, maxsize=(-1,) * k)
