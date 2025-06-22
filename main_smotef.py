from utils.evaluator_for_smotef import evaluate_smotef
from utils.preprocessor_for_smotef import preprocess_data_for_smotef
from Smotef.CFD_AA_Smurf import run_smotef

def main():
    reverse_mapping = preprocess_data_for_smotef("./data/HI-Small_Trans_Normalized.csv", "Smotef/data/AMLWorld")
    evaluate_smotef(reverse_mapping)

if __name__ == '__main__':
    main()
