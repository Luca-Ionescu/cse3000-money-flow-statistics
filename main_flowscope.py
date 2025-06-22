# This is a sample Python script.
from algorithms.flowscope_runner import run_flowscope_no_encoding
from utils.evaluator_for_flowscope import evaluate_f1_flowscope_accounts, evaluate_f1_flowscope
from utils.extractor_fraud_pattern import extract_all_patterns, extract_transactions
from utils.graph_plotter import plot_multipartite_graph_size
from utils.preprocessor_for_flowscope import split_dataset_for_flowscope_k_weighted
import pandas as pd


# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.


def main():
    tp_global = 0
    fp_global = 0
    for k in range(2, 6):
        split_dataset_for_flowscope_k_weighted("data/HI-Small_Trans_Normalized.csv", "output", k)
        tp, fp = evaluate_f1_flowscope(k)
        tp_global += tp
        fp_global += fp

    print(str(tp_global))
    print(str(fp_global))

    fraud_patterns = extract_all_patterns("./data/HI-Small_Patterns.txt")

    fn = len(fraud_patterns) - tp_global

    precision = tp_global / (tp_global + fp_global)
    recall = tp_global / (tp_global + fn)

    f1 = (2 * precision * recall) / (precision + recall)

    print("F1 score: " + str(f1))

    #plot_multipartite_graph_size(5)



if __name__ == '__main__':
    main()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
