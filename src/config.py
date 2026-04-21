from dataclasses import dataclass
import argparse


@dataclass
class Config:
    output_path: str
    input_path: str

    gen_shap: bool
    gen_confusion_matrix: bool

    num_studies: int
    shap_output_path: str
    confusion_matrix_output_path: str

    test_size: float
    random_state: int

    disable_smote: bool
    model_name: str

    verbose: bool
    show_warnings: bool

    max_rounds: int
    early_stopping_rounds: int
    num_folds: int


def get_config() -> Config:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="model",
        help="Directory where trained model artifacts and metadata will be saved",
    )

    parser.add_argument(
        "-i",
        "--input",
        type=str,
        default="data/training_data.csv",
        help="Path to the input CSV file containing training data",
    )

    parser.add_argument(
        "-s",
        "--shap",
        action="store_true",
        help="Generate and save SHAP feature importance visualization",
    )

    parser.add_argument(
        "-c",
        "--confusion-matrix",
        action="store_true",
        help="Generate and save confusion matrix visualization",
    )

    parser.add_argument(
        "--studies",
        type=int,
        default=1000,
        help="Number of Optuna trials for hyperparameter tuning",
    )

    parser.add_argument(
        "--shap-output",
        type=str,
        default="data/shap_visualized_data.png",
        help="File path to save the SHAP visualization image",
    )

    parser.add_argument(
        "--confusion-matrix-output",
        type=str,
        default="data/confusion_matrix.png",
        help="File path to save the confusion matrix image",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.25,
        help="Proportion of dataset to use as test set (e.g., 0.25 = 25%%)",
    )

    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility (affects data split and SMOTE)",
    )

    parser.add_argument(
        "--no-smote",
        action="store_true",
        help="Disable SMOTE oversampling on the training data",
    )

    parser.add_argument(
        "--model-name",
        type=str,
        default="model.ubj",
        help="Filename for saving the trained model",
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging during training and evaluation",
    )

    parser.add_argument(
        "--show-warnings",
        action="store_false",
        help="Disable suppression of warnings (show all warnings if set)",
    )

    parser.add_argument(
        "--max-rounds",
        type=int,
        default=1000,
        help="Maximum number of boosting rounds for XGBoost training",
    )

    parser.add_argument(
        "--early-stopping",
        type=int,
        default=20,
        help="Number of rounds with no improvement before early stopping",
    )

    parser.add_argument(
        "--num-folds",
        type=int,
        default=5,
        help="Number of folds for cross-validation (e.g., 5 = 5-fold CV)",
    )

    args = parser.parse_args()

    return Config(
        output_path=args.output,
        input_path=args.input,
        gen_shap=args.shap,
        gen_confusion_matrix=args.confusion_matrix,
        num_studies=args.studies,
        shap_output_path=args.shap_output,
        confusion_matrix_output_path=args.confusion_matrix_output,
        test_size=args.test_size,
        random_state=args.random_state,
        disable_smote=args.no_smote,
        model_name=args.model_name,
        verbose=args.verbose,
        show_warnings=args.show_warnings,
        max_rounds=args.max_rounds,
        early_stopping_rounds=args.early_stopping,
        num_folds=args.num_folds,
    )
