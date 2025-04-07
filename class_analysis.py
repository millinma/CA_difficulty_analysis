import pandas as pd
from sklearn.metrics import confusion_matrix
import yaml
from scipy.stats import spearmanr
import util_functions as uf
import numpy as np
import os

for task in uf.TASKS:
    task_output_dir = os.path.join(uf.ROOT_ANALYSIS_DIR, task)
    os.makedirs(task_output_dir, exist_ok=True)

    # Calculate mean difficulty per class
    difficulty_path = os.path.join(
        uf.SAMPLE_DIFFICULTY_DIR, uf.SCORING_FUNCTION + task + ".csv"
    )
    difficulty_df = pd.read_csv(difficulty_path)
    mean_difficulties_df = (
        difficulty_df.groupby(uf.TASK_INFO[task]["target_column"])["ranks"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    mean_difficulties_df.to_csv(
        os.path.join(task_output_dir, "mean_difficulty_values.csv"), index=False
    )

    # Calculate class-wise recalls of individual runs
    train_recalls_across_runs = []
    test_recalls_across_runs = []
    for run_specs in uf.TASK_SPECS[task]:
        run_folder = os.path.join(uf.ROOT_TRAINING_DIR, run_specs)

        # obtain labels from autrainer
        label_file = os.path.join(run_folder, "target_transform.yaml")
        with open(label_file, "r") as f:
            yaml_data = yaml.safe_load(f)
        label_names = yaml_data.get(
            "$autrainer.datasets.utils.target_transforms.label_encoder.LabelEncoder==0.4.0",
            {},
        ).get("labels", [])

        # test confusion matrix
        test_folder = os.path.join(run_folder, "_test")
        test_output_path = os.path.join(test_folder, "test_outputs.npy")
        test_target_path = os.path.join(test_folder, "test_targets.npy")
        test_outputs = np.load(test_output_path)
        test_targets = np.load(test_target_path)
        test_predictions = test_outputs.argmax(axis=1)
        test_cm = confusion_matrix(test_targets, test_predictions)
        test_cm_normalized = (
            test_cm.astype("float") / test_cm.sum(axis=1)[:, np.newaxis]
        )

        # train confusion matrix
        best_summary_file = os.path.join(run_folder, "_best", "best_results.yaml")
        with open(best_summary_file) as f:
            yaml_data = yaml.safe_load(f)
        best_epoch = yaml_data.get("best_iteration")
        train_folder = os.path.join(run_folder, "epoch_" + str(best_epoch))
        train_output_path = os.path.join(train_folder, "train_outputs.npy")
        train_target_path = os.path.join(train_folder, "train_targets.npy")
        train_outputs = np.load(train_output_path)
        train_targets = np.load(train_target_path)
        train_predictions = train_outputs.argmax(axis=1)
        train_cm = confusion_matrix(train_targets, train_predictions)
        train_cm_normalized = (
            train_cm.astype("float") / train_cm.sum(axis=1)[:, np.newaxis]
        )

        # Calculate recalls
        test_recalls = np.diagonal(test_cm_normalized)
        test_recalls_across_runs.append(test_recalls)
        train_recalls = np.diagonal(train_cm_normalized)
        train_recalls_across_runs.append(train_recalls)

    mean_train_recalls = np.mean(train_recalls_across_runs, axis=0)
    mean_test_recalls = np.mean(test_recalls_across_runs, axis=0)
    mean_difficulties = (
        mean_difficulties_df.set_index(uf.TASK_INFO[task]["target_column"])
        .loc[label_names, "ranks"]
        .tolist()
    )

    # calculated as one correlation from the mean recalls
    train_test_recall_correlation, _ = spearmanr(mean_train_recalls, mean_test_recalls)
    # calculated as the mean of run-wise correlation
    train_test_correlations = []
    for train_recalls in train_recalls_across_runs:
        train_test_correlation, _ = spearmanr(train_recalls, mean_test_recalls)
        train_test_correlations.append(train_test_correlation)
    mean_train_test_recall_single_run = np.mean(train_test_correlations)
    std_train_test_recall_single_run = np.std(train_test_correlations)
    # calculate correlation between mean difficulty and mean test recall
    difficulty_test_recall_correlation, _ = spearmanr(
        mean_difficulties, mean_test_recalls
    )
    correlation_df = pd.DataFrame(
        [
            ["Train - Test correlation", train_test_recall_correlation, 0],
            [
                "Train - Test correlation (over single train recalls)",
                mean_train_test_recall_single_run,
                std_train_test_recall_single_run,
            ],
            ["Difficulty Test", difficulty_test_recall_correlation, 0],
        ],
        columns=["Base", "Spearmanr", "STD"],
    )

    # save correlation results
    correlation_results_file = os.path.join(task_output_dir, "class_correlation.csv")
    correlation_df.to_csv(correlation_results_file, index=False)

    # save class-level recalls and difficulty
    label_names_written = [label_name.replace("_", " ") for label_name in label_names]
    class_df = pd.DataFrame(
        {
            "labels": label_names_written,
            "train_recall": mean_train_recalls,
            "test_recall": mean_test_recalls,
            "mean_difficulty": mean_difficulties,
        }
    )
    class_overview_file = os.path.join(task_output_dir, "class_overview.csv")
    class_df.to_csv(class_overview_file, index=False)
