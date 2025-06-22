from CFD_AA_Smurf import run_smotef
import pandas as pd
import networkx as nx

def evaluate_smotef(decode_mapping):
    valid_smurfs = run_smotef()

    flagged_accounts = set()

    df = (pd.read_csv("./data/HI-Small-Scatter-Gather-Transactions.csv", dtype={
        "From Bank": str,
        "To Bank": str
    }))

    G = nx.DiGraph()

    for _, row in df.iterrows():
        src = str(row["From Bank"]) + "+" + str(row["From Account"])
        dst = str(row["To Bank"]) + "+" + str(row["To Account"])
        amt = row["Amount Paid"]

        G.add_edge(src, dst, weight=amt)

    for pattern in valid_smurfs:
        for acc in pattern:
            flagged_accounts.add(decode_mapping.get(int(acc)))

    true_positives = []

    for u, v in G.edges():
        if u in flagged_accounts and v in flagged_accounts:
            true_positives.append((u, v))

    df = (pd.read_csv("./data/HI-Small_Trans_Normalized.csv", dtype={
        "From Bank": str,
        "To Bank": str
    }))

    G_full = nx.DiGraph()

    for _, row in df.iterrows():
        src = str(row["From Bank"]) + "+" + str(row["From Account"])
        dst = str(row["To Bank"]) + "+" + str(row["To Account"])
        amt = row["Amount Paid"]

        G_full.add_edge(src, dst, weight=amt)

    full_edges = []

    for u in flagged_accounts:
        for v in flagged_accounts:
            if u != v and G_full.has_edge(u, v):
                full_edges.append((u, v))

    print("Flagged edges: " + str(true_positives))

    tp = len(true_positives)
    fp = len(full_edges) - tp
    fn = len(G.edges()) - tp

    print("TP: " + str(tp))
    print("FP: " + str(fp))
    print("FN: " + str(fn))

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)

    print("Precision: " + str(precision))
    print("Recall: " + str(recall))

    f1 = (2 * precision * recall) / (precision + recall)

    print("F1: " + str(f1))

