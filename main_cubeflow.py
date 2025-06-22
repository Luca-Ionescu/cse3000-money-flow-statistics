from algorithms.cubeflow_runner import run_cubeflow
from utils.evaluator_for_cubeflow import evaluate_f1_cubeflow
from utils.extractor_fraud_pattern import visualize_data
from utils.preprocessor_for_cubeflow import split_dataset_for_cubeflow_k_weighted
from utils.preprocessor_for_flowscope import convert_currency_IBM_dataset


def main():
    k=2
    split_dataset_for_cubeflow_k_weighted("data/HI-Small_Trans_Normalized.csv", "output-cubeflow", k)
    evaluate_f1_cubeflow()

if __name__ == '__main__':
    main()