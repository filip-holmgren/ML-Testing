import warnings

from src.config import get_config
from src.data import load_data, split_data, apply_smote
from src.train import train_model
from src.eval import save_artifacts, evaluate_and_visualize


def main():
    config = get_config()

    if config.show_warnings:
        warnings.filterwarnings("ignore")

    X, y, category_maps = load_data(config.input_path)
    X_train, X_test, y_train, y_test = split_data(
        X, y, config.test_size, config.random_state
    )

    if config.disable_smote is False:
        X_train, y_train = apply_smote(X_train, y_train, config.random_state)

    model, best_thresh = train_model(X_train, y_train, X_test, y_test, config)

    save_artifacts(
        model, best_thresh, X, category_maps, config.output_path, config.model_name
    )

    evaluate_and_visualize(model, X_test, y_test, best_thresh, config)


if __name__ == "__main__":
    main()
