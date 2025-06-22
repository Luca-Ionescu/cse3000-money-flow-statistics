import pandas as pd
import networkx as nx
import os

def preprocess_data_for_smotef(input_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")

    df["HourBin"] = df["Timestamp"].dt.floor("1min")

    df["TimeBinIndex"] = df["HourBin"].rank(method="dense").astype(int) - 1

    # df = df.groupby(
    #     ["TimeBinIndex", "From Bank", "From Account", "To Bank", "To Account"],
    #     as_index=False
    # )["Amount Paid"].sum()

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt, date = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'], row["TimeBinIndex"])
        G.add_edge(src, dst, weight=amt, time=row['TimeBinIndex'])

    layer_edges = []

    mapping = {old_id: new_id for new_id, old_id in enumerate(sorted(G.nodes()))}
    reverse_mapping = {v: k for k, v in mapping.items()}

    for u, v in G.edges:
        amt = G[u][v]["weight"]
        time = G[u][v]["time"]
        layer_edges.append({'from': mapping.get(u), 'to': mapping.get(v), 'date': time, 'amount': int(amt)})

    df_edges = pd.DataFrame(layer_edges)
    filepath = os.path.join(output_dir, "data-for-smotef.csv")
    df_edges.to_csv(filepath, index=False, header=True)

    return reverse_mapping
