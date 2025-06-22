import pandas as pd
import os
import networkx as nx

def preprocess_dataset_for_denseflow(input_csv: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")

    df["HourBin"] = df["Timestamp"].dt.floor("24h")

    df["TimeBinIndex"] = df["HourBin"].rank(method="dense").astype(int) - 1

    df = df.groupby(
        ["TimeBinIndex", "From Bank", "From Account", "To Bank", "To Account"],
        as_index=False
    )["Amount Paid"].sum()

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt, date = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'], row["TimeBinIndex"])
        G.add_edge(src, dst, weight=amt, time=row['TimeBinIndex'])

    layer_edges = []
    for u, v in G.edges:

        amt = G[u][v]["weight"]
        time = G[u][v]["time"]
        layer_edges.append({'from_address': u, 'to_address': v, 'timestamp': time, 'amount': amt})

    df_edges = pd.DataFrame(layer_edges)
    filepath = os.path.join(output_dir, "data-for-denseflow.csv")
    df_edges.to_csv(filepath, index=False, header=True)

def extract_source_accounts_for_denseflow(file_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    laundering_graphs = []
    with open(file_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    current_graph = None

    for line in lines:
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            current_graph = nx.DiGraph()
            current_pattern_type = line.replace("BEGIN LAUNDERING ATTEMPT -", "").strip()
            in_pattern = True
            first_edge = None
        elif line.startswith("END LAUNDERING ATTEMPT"):
            if current_graph.number_of_nodes() > 0:
                if current_pattern_type.startswith("CYCLE") and first_edge:
                    source_nodes = {first_edge[0]}
                else:
                    source_nodes = {n for n, d in current_graph.in_degree() if d == 0}
                laundering_graphs.append({
                    "type": current_pattern_type,
                    "graph": current_graph,
                    "sources": source_nodes
                })
            current_graph = None
            current_pattern_type = ""
            in_pattern = False
        elif in_pattern and line and not line.startswith("DATE"):
            parts = line.split(",")
            if len(parts) >= 6:
                src = str(parts[1]) + "+" + str(parts[2])
                dst = str(parts[3]) + "+" + str(parts[4])
                amt = float(parts[5].strip())

                current_graph.add_edge(src, dst, amount=amt)
                if first_edge is None:
                    first_edge = (src, dst)

    with open(os.path.join(output_dir, "heist_sources.csv"), "w") as out:
        for g in laundering_graphs:
            for s in g["sources"]:
                out.write(f"{s},heist\n")
