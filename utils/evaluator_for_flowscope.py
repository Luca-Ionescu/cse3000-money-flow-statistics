import csv

import pandas as pd

from algorithms.flowscope_runner import run_flowscope
from utils.extractor_fraud_pattern import extract_patterns_with_length, extract_fraudulent_accounts_with_length, \
    extract_all_fraudulent_accounts, extract_all_patterns


def evaluate_f1_flowscope(k:int):
    res, int_to_id_per_layer = run_flowscope(k)
    flagged_transactions = []

    for entry in res:
        subgraph = entry[0]
        for layer_idx in range(0,k):
            src_layer = subgraph[layer_idx]
            dst_layer = subgraph[layer_idx + 1]

            for node_src in src_layer:
                for node_dst in dst_layer:
                    src = int_to_id_per_layer[layer_idx].get(node_src)
                    dst = int_to_id_per_layer[layer_idx + 1].get(node_dst)
                    flagged_transactions.append([src.split("+"), dst.split("+")])

    csv_transactions = set()
    with open("./data/HI-Small_Trans.csv", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            from_bank = row["From Bank"]
            from_account = row["From Account"]
            to_bank = row["To Bank"]
            to_account = row["To Account"]
            csv_transactions.add((from_bank, from_account, to_bank, to_account))

    filtered_flagged_transactions = [
        pair for pair in flagged_transactions
        if (pair[0][0], pair[0][1], pair[1][0], pair[1][1]) in csv_transactions
    ]

    df = pd.read_csv("./data/HI-Small_Trans_Normalized.csv", dtype={
        "From Bank": str,
        "To Bank": str
    })

    df = df.groupby(
        ["From Bank", "From Account", "To Bank", "To Account"],
        as_index=False
    )["Amount Paid"].sum()

    full_transactions = []
    for trans in filtered_flagged_transactions:

        full_trans = df[
            (df["From Bank"] == trans[0][0]) & (df["From Account"] == trans[0][1]) & (df["To Bank"] == trans[1][0]) & (
                        df["To Account"] == trans[1][1])]
        print(full_trans)
        full_transactions.append(full_trans)

    df["tuple_col"] = list(zip(df["From Bank"], df["From Account"], df["To Bank"], df["To Account"]))

    all_rows = pd.concat(full_transactions, ignore_index=True)
    all_rows.to_csv(f"./auxiliary/found-patterns-{k}.csv", index=False)


    df_filtered = df[df["tuple_col"].isin(filtered_flagged_transactions)]

    df = df_filtered.drop(columns=["tuple_col"])

    print(df)

    fraud_patterns = extract_all_patterns("./data/HI-Small_Patterns.txt")
    fraud_patterns = [trans for pair in fraud_patterns for trans in pair]

    print(filtered_flagged_transactions)
    #print(fraud_patterns)

    tp = 0
    fp = 0
    fn = 0

    correctly_flagged = set()

    for fraud_pattern in fraud_patterns:
        details = fraud_pattern.split(",")
        for flagged in filtered_flagged_transactions:

            if(details[1] == flagged[0][0] and details[2] == flagged[0][1] and details[3] == flagged[1][0] and details[4] == flagged[1][1]):
                correctly_flagged.add(str(flagged))
                tp = tp + 1

    fp = len(filtered_flagged_transactions) - len(correctly_flagged)

    print("TP: " + str(tp))
    print("FP: " + str(fp))
    print("FN: " + str(fn))

    return tp, fp

def evaluate_f1_flowscope_accounts(k: int):
    res, int_to_id_per_layer = run_flowscope(k)
    flagged_accounts = []

    for entry in res:
        subgraph = entry[0]
        for idx, layer in enumerate(subgraph):
            for node in layer:
                idents = int_to_id_per_layer[idx].get(node).split("+")
                flagged_accounts.append((idents[0], idents[1]))


    fraudulent_accounts = extract_all_fraudulent_accounts("./data/HI-Small_Patterns.txt")

    tp = 0
    fp = 0
    fn = 0

    print("Flagged: " + str(flagged_accounts))
    print("Real: " + str(fraudulent_accounts))

    for account in flagged_accounts:
        if account in fraudulent_accounts:
            tp = tp + 1
        else:
            fp = fp + 1

    fn = len(fraudulent_accounts) - tp

    print("TP: " + str(tp))
    print("FP: " + str(fp))
    print("FN: " + str(fn))