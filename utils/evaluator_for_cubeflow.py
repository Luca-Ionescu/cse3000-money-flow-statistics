from algorithms.cubeflow_runner import run_cubeflow
import pandas as pd
import networkx as nx

from utils.preprocessor_for_flowscope import all_simple_paths_of_length_k


def evaluate_f1_cubeflow():
    res, int_to_id = run_cubeflow(2)


    df = (pd.read_csv("./data/HI-Small_Trans_Normalized.csv", dtype={
        "From Bank": str,
        "To Bank": str
    }))

    df = df.groupby(
        ["From Bank", "From Account", "To Bank", "To Account"],
        as_index=False
    )["Amount Paid"].sum()

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'])
        G.add_edge(src, dst, weight=amt)

    flagged_edges = []

    for idx, block in enumerate(res):
        block_data = block[0]
        sources = block_data[0]
        middles = block_data[1]
        destinations = block_data[2]

        print("Analyzing block " + str(idx))

        for y in middles:
            mid = int_to_id.get(y)
            for x in sources:
                src = int_to_id.get(x)
                if G.has_edge(src, mid):
                    flagged_edges.append((src, mid))
                    print(str(src) + " -- " + str(G[src][mid]['weight']) + " --> " + str(mid))

            for z in destinations:
                dst = int_to_id.get(z)
                if G.has_edge(mid, dst):
                    flagged_edges.append((mid, dst))
                    print(str(mid) + " -- " + str(G[mid][dst]['weight']) + " --> " + str(dst))

    df = (pd.read_csv("./data/HI-Small_Laundering_Transactions_Normalized.csv", dtype={
        "From Bank": str,
        "To Bank": str
    }))

    df = df.groupby(
        ["From Bank", "From Account", "To Bank", "To Account"],
        as_index=False
    )["Amount Paid"].sum()

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'])
        G.add_edge(src, dst, weight=amt)

    all_edges = set()

    for node in G.nodes:
        paths = all_simple_paths_of_length_k(G, node, 2)
        for path in paths:
            all_edges.add((path[0], path[1]))
            all_edges.add((path[1], path[2]))

    tp = 0
    fp = 0
    fn = 0

    for flagged_edge in flagged_edges:
        if flagged_edge in all_edges:
            tp+=1

    fp = len(flagged_edges) - tp
    fn = len(all_edges) - tp

    print("TP: " + str(tp))
    print("FP: " + str(fp))
    print("FN: " + str(fn))