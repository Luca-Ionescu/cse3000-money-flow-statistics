from algorithms.denseflow_runner import prepare_case_data, run_denseflow, compute_neighbourhood_sources, \
    percentage_sources_under_k_descendants
from utils.evaluator_for_denseflow import evaluate_denseflow_accounts
from utils.preprocessor_for_denseflow import preprocess_dataset_for_denseflow, extract_source_accounts_for_denseflow


def main():
    #preprocess_dataset_for_denseflow("data/HI-Small_Trans_Normalized.csv", "output-denseflow")
    #extract_source_accounts_for_denseflow("data/HI-Small_Patterns.txt", "output-denseflow")

    #prepare_case_data("AMLWorld", "data-for-denseflow.csv", "heist_sources.csv")

    #
    # compute_neighbourhood_sources("data/HI-Small_Trans_Normalized.csv", "DenseFlow_Tool/inputData/AML/AMLWorld/accounts-hacker.csv", "auxiliary/descendants-of-laundering-sources.csv")
    #
    # percentage_sources_under_k_descendants("auxiliary/descendants-of-laundering-sources.csv", 15)

   # run_denseflow("AMLWorld", k=6, levels=4)
    evaluate_denseflow_accounts(6, "./data/HI-Small_Laundering_Transactions.csv")

if __name__ == '__main__':
    main()