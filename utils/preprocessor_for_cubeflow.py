import os
from collections import defaultdict

import pandas as pd
import networkx as nx

from utils.preprocessor_for_flowscope import all_simple_paths_of_length_k


def split_dataset_for_cubeflow_k_weighted(input_csv: str, output_dir: str, k: int):
    os.makedirs(output_dir, exist_ok=True)

    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))

    df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%Y/%m/%d %H:%M")

    # Floor to the hour
    df["HourBin"] = df["Timestamp"].dt.floor("24h")
    #
    # non_laundering = df[df["Is Laundering"] == 0]
    # laundering = df[df["Is Laundering"] == 1]
    #
    # #Sample 50% of the non-laundering rows
    # non_laundering_sampled = non_laundering.sample(frac=0.1, random_state=42)
    #
    # # Combine the filtered non-laundering rows with all laundering rows
    # df = pd.concat([non_laundering_sampled, laundering], ignore_index=True)

    df["TimeBinIndex"] = df["HourBin"].rank(method="dense").astype(int) - 1

    df = df.groupby(
        ["TimeBinIndex", "From Bank", "From Account", "To Bank", "To Account"],
        as_index=False
    )["Amount Paid"].sum()

    #threshold = df["Amount Paid"].quantile(0.99)

    #df = df[df["Amount Paid"] <= threshold]

    # amount_threshold = df["Amount Paid"].quantile(0.80)
    # df = df[df["Amount Paid"] <= amount_threshold]

    #print(df)

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt, date = (
        (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
        row['Amount Paid'], row["TimeBinIndex"])

        # date_time_data = date.split(" ")
        #
        # date = date_time_data[0]
        #
        # date_data = date.split("-")
        # day = int(date_data[2])
        #
        # time = date_time_data[1]
        #
        # time_data = time.split(":")
        # hour = int(time_data[0])
        # minute = int(time_data[1])

        G.add_edge(src, dst, weight=amt, time = row['TimeBinIndex'])

    indeg = defaultdict(int)
    outdeg = defaultdict(int)
    for u, v in G.edges:
        outdeg[u] += G[u][v]["weight"]
        indeg[v] += G[u][v]["weight"]

    source = set()
    middle = set()
    sink = set()

    threshold = 10
    #
    # all_paths = []
    #
    for node in G.nodes:
    #     paths = all_simple_paths_of_length_k(G, node, 2)
    #     all_paths.extend(paths)

        if outdeg[node]/(indeg[node] + 0.00001) >= threshold:
            source.add(node)
        elif indeg[node]/(outdeg[node] + 0.00001) >= threshold:
            sink.add(node)
        else:
            middle.add(node)

    # for path in all_paths:
    #     source.add(path[0])
    #     middle.add(path[1])
    #     sink.add(path[1])

    print("Source: " + str(len(source)))
    print("Middle: " + str(len(middle)))
    print("Sink: " + str(len(sink)))

    # middle_filtered = set()
    # for y in middle:
    #     if 0.8 <= indeg[y] / (outdeg[y] + 1e-5) <= 1.2:
    #         middle_filtered.add(y)
    #
    # middle = middle_filtered

    # edges_source_middle = [(u, v) for u, v in G.edges if u in source and v in middle]
    # edges_middle_sink = [(u, v) for u, v in G.edges if u in middle and v in sink]

    matched_edges_source_middle = []
    matched_edges_middle_sink = []

    for y in middle:
        for u in G.predecessors(y):
            for v in G.successors(y):
                if (u in source and v in sink
                        #and 0.8 < indeg[y]/(outdeg[y] + 1e-5) < 1.2 and abs(G[u][y]["time"] - G[y][v]["time"]) <= 2
                        ):

                    # only accept if time is close and amount is close
                    matched_edges_source_middle.append((u, y))
                    matched_edges_middle_sink.append((y, v))


    # all_paths = []
    # for node in G.nodes:
    #     node_paths = all_simple_paths_of_length_k(G, node, k)
    #     all_paths.extend(node_paths)
    # if not all_paths:
    #     print(f"No paths of length {k} found.")
    #     return
    #
    #
    # filtered_paths = []
    #
    # print(len(all_paths))
    # for path in all_paths:
    #     v1 = path[0]
    #     v2 = path[1]
    #     v3 = path[2]
    #
    #     # print(str(G[v2][v3]["time"] - G[v1][v2]["time"]) + " " + str(G[v2][v3]["weight"] / G[v1][v2]["weight"]))
    #
    #
    #     if (2 >= G[v2][v3]["time"] - G[v1][v2]["time"] >= 0) and (0.8 < indeg[v2] / outdeg[v2] < 1.2) == True:
    #         filtered_paths.append(path)

    edge_timebins = {}

    for _, row in df.iterrows():
        key = (row["From Bank"], row["From Account"], row["To Bank"], row["To Account"])
        edge_timebins.setdefault(key, []).append(row["TimeBinIndex"])

    #filtered_paths = []
    # for path in all_paths:
    #     u1, u2, u3 = path[:3]
    #
    #     bank1, acc1 = u1.split("+")
    #     bank2, acc2 = u2.split("+")
    #     bank3, acc3 = u3.split("+")
    #
    #     key1 = (bank1, acc1, bank2, acc2)
    #     key2 = (bank2, acc2, bank3, acc3)
    #
    #     if key1 in edge_timebins and key2 in edge_timebins:
    #         times1 = set(edge_timebins[key1])
    #         times2 = set(edge_timebins[key2])
    #
    #         # Check if any t1 - t2 in (-1, 0)
    #         if any((t1 - t2) in (-2, -1, 0, 1, 2) for t1 in times1 for t2 in times2):
    #             filtered_paths.append(path)

    layer_edges = []
    for idx, edge in enumerate(matched_edges_source_middle):
        u = edge[0]
        v = edge[1]

        amt = G[u][v]["weight"]
        time = G[u][v]["time"]
        layer_edges.append({'row_id': idx, '0': u, '1': v, '2':time, '3':amt})

    df_edges = pd.DataFrame(layer_edges)
    filepath = os.path.join(output_dir, "fs1.csv")
    df_edges.to_csv(filepath, index=False)
    print(f"[Layer 1] Wrote {len(df_edges)} weighted edges to {filepath}")

    layer_edges = []
    for idx, edge in enumerate(matched_edges_middle_sink):
        u = edge[0]
        v = edge[1]

        amt = G[u][v]["weight"]
        time = G[u][v]["time"]
        layer_edges.append({'row_id': idx, '0': v, '1': u, '2': time, '3': amt})

    df_edges = pd.DataFrame(layer_edges)
    filepath = os.path.join(output_dir, "fs2.csv")
    df_edges.to_csv(filepath, index=False)
    print(f"[Layer 2] Wrote {len(df_edges)} weighted edges to {filepath}")

    # crt = 0
    # for idx, layer in enumerate(range(k)):
    #     layer_edges = []
    #     for p in filtered_paths:
    #         u, v = p[layer], p[layer + 1]
    #         if G.has_edge(u, v):
    #             amt = G[u][v]['weight']
    #             time = G[u][v]['time']
    #             layer_edges.append({'row id': crt, '0': u, '1': v, '2': time,'3': amt})
    #             crt = crt + 1
    #
    #     df_edges = pd.DataFrame(layer_edges)
    #     filepath = os.path.join(output_dir, f"fs{layer + 1}.csv")
    #     df_edges.to_csv(filepath, index=False)
    #print(f"[Layer {layer + 1}] Wrote {len(df_edges)} weighted edges to {filepath}")