import csv

def extract_transactions(input_file, output_csv):
    with open(input_file, 'r') as infile:
        lines = infile.readlines()

    transaction_lines = [
        line.strip() for line in lines
        if "BEGIN" not in line and "END" not in line and line.strip().strip('"') != ''
    ]

    with open(output_csv, 'w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow([
            "Timestamp", "From Bank", "From Account", "To Bank", "To Account",
            "Amount Received", "Receiving Currency", "Amount Paid", "Payment Currency",
            "Payment Format", "Is Laundering"
        ])
        for line in transaction_lines:
            writer.writerow(line.split(','))


def extract_patterns_with_length(k:int, filepath:str):
    file = open(filepath)
    lines = [line.strip() for line in file]

    blocks = []
    current_block = []
    inside_block = False

    for line in lines:
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            current_block = [line]
            inside_block = True
        elif line.startswith("END LAUNDERING ATTEMPT"):
            current_block.append(line)
            transaction_lines = [l for l in current_block if not l.startswith("BEGIN") and not l.startswith("END")]
            if len(transaction_lines) == k:
                blocks.append(transaction_lines)
            inside_block = False
        elif inside_block:
            current_block.append(line)

    return blocks

def extract_all_patterns(filepath: str):
    file = open(filepath)
    lines = [line.strip() for line in file]

    blocks = []
    current_block = []
    inside_block = False

    for line in lines:
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            current_block = [line]
            inside_block = True
        elif line.startswith("END LAUNDERING ATTEMPT"):
            current_block.append(line)
            transaction_lines = [l for l in current_block if not l.startswith("BEGIN") and not l.startswith("END")]
            blocks.append(transaction_lines)
            inside_block = False
        elif inside_block:
            current_block.append(line)


    return blocks

def extract_all_fraudulent_accounts(filepath: str):
    nodes = set()

    blocks = extract_all_patterns(filepath)
    transactions = [trans for pair in blocks for trans in pair]

    for transaction in transactions:
        details = transaction.split(",")
        nodes.add((details[1], details[2]))
        nodes.add((details[3], details[4]))

    return nodes


def extract_fraudulent_accounts_with_length(k: int, filepath: str):
    nodes = set()

    blocks = extract_patterns_with_length(k, filepath)
    transactions = [trans for pair in blocks for trans in pair]

    for transaction in transactions:
        details = transaction.split(",")
        nodes.add((details[1], details[2]))
        nodes.add((details[3], details[4]))

    return nodes

def extract_scatter_gather(input_path, output_path):
    with open(input_path, "r") as infile:
        lines = infile.readlines()

    inside_scatter_gather = False
    scatter_gather_transactions = []

    for line in lines:
        line_upper = line.upper()
        if "BEGIN LAUNDERING ATTEMPT - SCATTER-GATHER" in line_upper:
            inside_scatter_gather = True
            continue
        elif "END LAUNDERING ATTEMPT - SCATTER-GATHER" in line_upper:
            inside_scatter_gather = False
            continue
        if inside_scatter_gather and line.strip():
            scatter_gather_transactions.append(line.strip())

    with open(output_path, "w") as outfile:
        for transaction in scatter_gather_transactions:
            outfile.write(transaction + "\n")

if __name__ == '__main__':
    extract_scatter_gather("../data/HI-Small_Patterns.txt", "../data/HI-Small-Scatter-Gather-Transactions.csv")

def visualize_data():
    import pandas as pd

    file_path = "./data/HI-Small_Trans_Normalized.csv"
    chunk_size = 100_000

    unique_sources = set()
    unique_sinks = set()
    mid_account_counts = {}
    time_bins = {}

    total_txns = 0

    for chunk in pd.read_csv(file_path, chunksize=chunk_size, parse_dates=["Timestamp"], date_parser=lambda col: pd.to_datetime(col, format="%Y/%m/%d %H:%M")):

        total_txns += len(chunk)

        chunk["From"] = chunk["From Bank"].astype(str) + "+" + chunk["From Account"].astype(str)
        chunk["To"] = chunk["To Bank"].astype(str) + "+" + chunk["To Account"].astype(str)

        unique_sources.update(chunk["From"].unique())
        unique_sinks.update(chunk["To"].unique())

        for acc in chunk["To"]:
            mid_account_counts[acc] = mid_account_counts.get(acc, 0) + 1

        chunk["Hour"] = chunk["Timestamp"].dt.floor("1h")
        for t in chunk["Hour"]:
            time_bins[t] = time_bins.get(t, 0) + 1

    print("Total transactions:", total_txns)
    print("Unique source accounts:", len(unique_sources))
    print("Unique sink accounts:", len(unique_sinks))
    print("Accounts appearing more than once as mid:", sum(1 for v in mid_account_counts.values() if v > 1))
    print("Total time bins:", len(time_bins))
