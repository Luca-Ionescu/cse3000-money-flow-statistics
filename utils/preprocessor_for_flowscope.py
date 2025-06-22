import pandas as pd
import networkx as nx
import os


def all_simple_paths_of_length_k(G, source, k):
    paths = []

    def dfs(current_path):
        if len(current_path) == k + 1:
            paths.append(list(current_path))
            return
        last_node = current_path[-1]
        for neighbor in G.successors(last_node):
            if neighbor not in current_path:
                dfs(current_path + [neighbor])

    dfs([source])
    return paths

def split_dataset_for_flowscope_k_weighted(input_csv: str, output_dir: str, k: int, max_rows: int = 1000):
    os.makedirs(output_dir, exist_ok=True)

    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))



    currency_conversion = pd.read_csv("./auxiliary/currencies-conversion.csv")

    # #df= df[(df["Amount Paid"] < 800000.0)]
    # non_laundering = df[df["Is Laundering"] == 0]
    # laundering = df[df["Is Laundering"] == 1]
    #
    # #Sample 50% of the non-laundering rows
    # non_laundering_sampled = non_laundering.sample(frac=0.1, random_state=42)
    #
    # # Combine the filtered non-laundering rows with all laundering rows
    # df = pd.concat([non_laundering_sampled, laundering], ignore_index=True)

    df = df.groupby(
        ["From Bank", "From Account", "To Bank", "To Account"],
        as_index=False
    )["Amount Paid"].sum()

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt = ((str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
                         row['Amount Paid'])
        G.add_edge(src, dst, weight=amt)

    all_paths = []
    for node in G.nodes:
        node_paths = all_simple_paths_of_length_k(G, node, k)
        all_paths.extend(node_paths)
    if not all_paths:
        print(f"No paths of length {k} found.")
        return

    crt = 0
    for idx, layer in enumerate(range(k)):
        layer_edges = []
        for p in all_paths:
            u, v = p[layer], p[layer + 1]
            if G.has_edge(u, v):
                amt = G[u][v]['weight']
                layer_edges.append({'row id': crt, '0': u, '1': v, '2': amt})
                crt = crt + 1

        df_edges = pd.DataFrame(layer_edges).drop_duplicates()
        filepath = os.path.join(output_dir, f"fs{layer + 1}.csv")
        df_edges.to_csv(filepath, index=False)
        print(f"[Layer {layer + 1}] Wrote {len(df_edges)} weighted edges to {filepath}")

def extract_all_currencies(input_csv):
    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))

    unique_currencies = df["Payment Currency"].dropna().unique()

    unique_df = pd.DataFrame(unique_currencies, columns=["Payment Currency"])

    unique_df.to_csv("./auxiliary/currencies.csv", index=False)

def convert_currency_IBM_dataset(input_csv):
    df = (pd.read_csv(f"./data/{input_csv}.csv", dtype={
        "From Bank": str,
        "To Bank": str
    }))

    currency_conversion = pd.read_csv("./auxiliary/currencies-conversion.csv")

    df = df.merge(currency_conversion, how="left", left_on="Payment Currency", right_on="Payment Currency")

    df["Amount Paid"] = df["Amount Paid"] * df["Conversion"]

    df.to_csv(f"./data/{input_csv}_Normalized.csv", index=False)


