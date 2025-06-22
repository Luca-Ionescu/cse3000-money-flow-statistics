import pandas as pd
import networkx as nx

def evaluate_denseflow_accounts(k: int, input_csv: str):
    path = "./DenseFlow_Tool/inputData/AML/AMLWorld/AMLWorld_out/"

    excel_files = [path + f"AMLWorld_k_{k}_level_{i}.xlsx" for i in range(1)]

    dfs = [pd.read_excel(f, usecols=["holo_heist"]) for f in excel_files]
    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df = combined_df.drop_duplicates().reset_index(drop=True)

    node_list = combined_df['holo_heist'].astype(str).tolist()

    df = (pd.read_csv(input_csv, dtype={
        "From Bank": str,
        "To Bank": str
    }))

    fraud_accounts = set()

    for _, row in df.iterrows():
        acc1 = str(row["From Bank"]) + "+" + str(row["From Account"])
        acc2 = str(row["To Bank"]) + "+" + str(row["To Account"])

        fraud_accounts.add(acc1)
        fraud_accounts.add(acc2)

    true_positives = [acc for acc in fraud_accounts if acc in node_list]

    tp = len(true_positives)
    fp = len(node_list) - len(true_positives)
    fn = len(fraud_accounts) - len(true_positives)

    print("TP: " + str(tp))
    print("FP: " + str(fp))
    print("FN: " + str(fn))




