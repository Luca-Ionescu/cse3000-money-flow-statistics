from Code.Datatran_1 import csvtodata
from Code.holodatatran import output_holo
from Code.myHoloscope import output_holo as output_holo_runner
import pandas as pd
import os
import spartan as st
import networkx as nx
import matplotlib.pyplot as plt

from Code.myMaxflow import output_flow_add, read_g, build_g


def prepare_case_data(case_name: str, tx_csv: str, hacker_csv: str):
    """Prepare transaction and hacker source files in the expected directory structure."""
    base_dir = f'DenseFlow_Tool/inputData/AML/{case_name}/'
    os.makedirs(base_dir, exist_ok=True)

    # Prepare transaction data
    df = pd.read_csv(base_dir + tx_csv)
    df.rename(columns={
        'from_address': 'from',
        'to_address': 'to',
        'timestamp': 'timeStamp',
        'amount': 'value'
    }, inplace=True)
    df['hash'] = ['tx_' + str(i) for i in range(len(df))]  # create dummy transaction IDs
    df.to_csv(os.path.join(base_dir, 'all-normal-tx.csv'), index=False)

    # Prepare hacker/source addresses
    hackers = pd.read_csv(base_dir + hacker_csv)
    hackers.to_csv(os.path.join(base_dir, 'accounts-hacker.csv'), index=False)

def run_denseflow(case_name: str, k: int = 0, levels: int = 4):
    """Run the full DenseFlow pipeline for a given case."""

    print(f"[1] Mapping Ethereum addresses to integer IDs...")
    csvtodata(case_name)

    print(f"[2] Computing suspiciousness metrics for {case_name}...")
    output_holo(case_name)

    print(f"[3] Constructing the tensor graph for FlowScope...")
    tensor_path = f'DenseFlow_Tool/inputData/AML/{case_name}/{case_name}_holo.csv'
    tensor_data = st.loadTensor(path=tensor_path, header=None)

    # Clean and format tensor data
    tensor_data.data = tensor_data.data.drop(columns=[0])
    tensor_data.data = tensor_data.data.drop([0])
    tensor_data.data.columns = [0, 1, 2, 3, 4]
    tensor_data.data[2] = tensor_data.data[2].str.replace('/', '-')



    tensor_data.data[2] = pd.to_datetime(tensor_data.data[2]).dt.strftime('%Y-%m-%d')
    unique_dates = {date: idx for idx, date in enumerate(sorted(tensor_data.data[2].unique()))}
    tensor_data.data[2] = tensor_data.data[2].map(unique_dates).astype(int)

    # Step 2: Make sure other coordinate columns are also integers
    tensor_data.data[0] = tensor_data.data[0].astype(int)
    tensor_data.data[1] = tensor_data.data[1].astype(int)

    # Tensor from cleaned data
    stensor = tensor_data.toSTensor(hasvalue=True)
    #stensor = tensor_data.toSTensor(hasvalue=True, mappers={3: st.TimeMapper(timeformat='%Y-%m-%d')})
    graph = st.Graph(stensor, bipartite=True, weighted=True, modet=2)

    hs = st.HoloScope(graph)  # used as FlowScope
    print(f"[4] Running FlowScope block detection...")
    output_holo_runner("AMLWorld", k, hs)

    print(f"[5] Preparing maxflow graph for fund tracing...")
    gpath = f'./Maxflow_graph/{case_name}/'
    if not os.path.exists(gpath + 'start_nodes.npy'):
        start_nodes, end_nodes, capacities, nodenum = build_g(case_name)
    else:
        start_nodes, end_nodes, capacities, nodenum = read_g(case_name)

    # Load hacker source node
    hacker_file = f'DenseFlow_Tool/inputData/AML/{case_name}/accounts-hacker.csv'
    source_addrs = pd.read_csv(hacker_file)['address'].tolist()
    source_id = source_addrs[2]  # assumes only 1 source

    print(f"[6] Performing max-flow tracing from source node {source_id}...")
    for level in range(levels):
        output_flow_add(case_name, k, level, source_id)
        print(f"  → Max-flow paths computed for block {level}")

def compute_neighbourhood_sources(input_csv: str, sources_csv: str, output_csv: str):
    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'])
        G.add_edge(src, dst, weight=amt)

    df = pd.read_csv(sources_csv, usecols=["address"])

    descendants = []

    for addr in df["address"]:
        descendants.append((addr, len(nx.descendants(G, addr))))

    df_desc = pd.DataFrame(descendants, columns=["id", "descendants"])
    df_desc.to_csv(output_csv, index=False)

    # Compute average number of descendants
    avg_descendants = df_desc["descendants"].mean()

    print(avg_descendants)

    return df_desc, avg_descendants

def percentage_sources_under_k_descendants(file_path: str, k : int):
    df = pd.read_csv(file_path)

    # Compute percentage
    percentage_below_k = (df["descendants"] < k).mean() * 100

    print(percentage_below_k)

def get_statistics_dense_subgraph(input_csv: str, results_csv: str):
    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'])
        G.add_edge(src, dst, weight=amt)

    df = pd.read_excel(results_csv, usecols=["holo_heist"])
    ids = df["holo_heist"].tolist()

    subgraph = G.subgraph(ids).copy()

    num_nodes = subgraph.number_of_nodes()

    avg_in_degree = sum(dict(subgraph.in_degree()).values()) / num_nodes
    avg_out_degree = sum(dict(subgraph.out_degree()).values()) / num_nodes

    print("DenseFlow subgraph: ")
    print("AVG indegree: " + str(avg_in_degree))
    print("AVG outdegree: " + str(avg_out_degree))

def get_statistic_known_laundering_edges(input_csv: str):
    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'])
        G.add_edge(src, dst, weight=amt)

    weights = nx.get_edge_attributes(G, 'weight').values()

    num_nodes = G.number_of_nodes()
    if num_nodes == 0:
        return 0.0, 0.0

    avg_in_degree = sum(dict(G.in_degree()).values()) / num_nodes
    avg_out_degree = sum(dict(G.out_degree()).values()) / num_nodes

    print("Laundering graph: ")
    print("AVG indegree: " + str(avg_in_degree))
    print("AVG outdegree: " + str(avg_out_degree))


def get_average_edge_weight_in_entire_graph_and_laundering_edges():
    df = (pd.read_csv("../data/HI-Small_Trans_Normalized.csv", dtype={
        "From Bank": str,
        "To Bank": str
    }))

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'])
        G.add_edge(src, dst, weight=amt)

    weights = nx.get_edge_attributes(G, 'weight').values()

    print("Average weight in entire dataset: " + str(sum(weights)/len(weights)))

    df = (pd.read_csv("../data/HI-Small_Laundering_Transactions_Normalized.csv", dtype={
        "From Bank": str,
        "To Bank": str
    }))

    G = nx.DiGraph()
    for _, row in df.iterrows():
        src, dst, amt = (
            (str(row['From Bank']) + "+" + row['From Account']), (str(row['To Bank']) + "+" + row['To Account']),
            row['Amount Paid'])
        G.add_edge(src, dst, weight=amt)

    weights = nx.get_edge_attributes(G, 'weight').values()

    print("Average weight in laundering dataset: " +  str(sum(weights)/len(weights)))

def plot_time_distribution_laundering_edges():
    laundering_df = pd.read_csv("../data/HI-Small_Laundering_Transactions_Normalized.csv", dtype={
        "From Bank": str,
        "To Bank": str
    })

    laundering_df['Date'] = pd.to_datetime(laundering_df['Timestamp'], format='%Y/%m/%d %H:%M').dt.date

    # Count the number of laundering transactions per day
    laundering_per_day = laundering_df['Date'].value_counts().sort_index()

    # Plot the histogram
    plt.figure(figsize=(12, 6))
    laundering_per_day.plot(kind='bar')
    plt.xlabel("Date")
    plt.ylabel("Number of Laundering Transactions")
    plt.title("Histogram of Laundering Transactions per Day (HI-Small)")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.grid(True)
    plt.show()

def main():
    # get_statistics_dense_subgraph("../data/HI-Small_Trans_Normalized.csv", "../DenseFlow_Tool/inputData/AML/AMLWorld/AMLWorld_out/AMLWorld_k_5_level_0.xlsx")
    # get_statistic_known_laundering_edges("../data/HI-Small_Laundering_Transactions_Normalized.csv")

    # get_average_edge_weight_in_entire_graph_and_laundering_edges()

    plot_time_distribution_laundering_edges()

if __name__ == '__main__':
    main()


