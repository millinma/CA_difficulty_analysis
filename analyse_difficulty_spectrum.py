from autrainer.serving import Inference
from torchcam.methods import GradCAM
import pandas as pd
import matplotlib.pyplot as plt
import shutil
import torch.nn.functional as F
import matplotlib
import yaml
import util_functions as uf
import numpy as np
import os

matplotlib.use("Agg")
plt.ioff()


model = "Cnn10"
n_samples_per_class = 10
layer = "conv_block4.conv2"


for task in uf.TASKS:
    task_output_dir = os.path.join(uf.ROOT_ANALYSIS_DIR, task)
    os.makedirs(task_output_dir, exist_ok=True)

    # load difficulty dir
    difficulty_path = os.path.join(
        uf.SAMPLE_DIFFICULTY_DIR, uf.SCORING_FUNCTION + task + ".csv"
    )
    difficulty_df = pd.read_csv(difficulty_path)

    # create subsebts at the extreme ends of the difficulty spectrum
    easy_dir = os.path.join(task_output_dir, "easy")
    hard_dir = os.path.join(task_output_dir, "hard")
    medium_dir = os.path.join(task_output_dir, "medium")
    os.makedirs(easy_dir, exist_ok=True)
    os.makedirs(medium_dir, exist_ok=True)
    os.makedirs(hard_dir, exist_ok=True)
    low_df = (
        difficulty_df.groupby(uf.TASK_INFO[task]["target_column"])
        .apply(lambda x: x.nsmallest(n_samples_per_class, "ranks"))
        .reset_index(drop=True)
    )
    high_df = (
        difficulty_df.groupby(uf.TASK_INFO[task]["target_column"])
        .apply(lambda x: x.nlargest(n_samples_per_class, "ranks"))
        .reset_index(drop=True)
    )
    medium_df = (
        difficulty_df.groupby(uf.TASK_INFO[task]["target_column"])
        .apply(
            lambda x: uf.get_medium_values(x.sort_values("ranks"), n_samples_per_class)
        )
        .reset_index(drop=True)
    )
    for i, run_specs in enumerate(uf.TASK_SPECS[task]):
        # make sure we only use the correct specified model
        if run_specs.split("_")[1] != model:
            continue
        if i == 0:
            run_folder = os.path.join(uf.ROOT_TRAINING_DIR, run_specs)
            label_file = os.path.join(run_folder, "target_transform.yaml")
            with open(label_file, "r") as f:
                yaml_data = yaml.safe_load(f)
            label_names = yaml_data.get(
                "$autrainer.datasets.utils.target_transforms.label_encoder.LabelEncoder==0.4.0",
                {},
            ).get("labels", [])
        # load model for inference
        model_path = os.path.join(uf.ROOT_TRAINING_DIR, run_specs)
        inference = Inference(
            model_path=model_path,
        )
        inference.model.eval()
        for difficulty_dir, difficulty_df in zip(
            [easy_dir, medium_dir, hard_dir], [low_df, medium_df, high_df]
        ):
            for index, row in difficulty_df.iterrows():
                # get file info and location
                filename = row[uf.TASK_INFO[task]["index_column"]]
                filepath = os.path.join(
                    uf.TASK_INFO[task]["data_dir"], "default", filename
                )
                label = row[uf.TASK_INFO[task]["target_column"]]

                # target path under which spectrogram, explanation, and raw file is saved
                out_path_no_extension = os.path.join(
                    difficulty_dir,
                    label
                    + "-"
                    + str(index).zfill(2)
                    + "-"
                    + os.path.basename(filename).replace(".wav", ""),
                )
                # copy raw file
                if i == 0:
                    shutil.copyfile(filepath, out_path_no_extension + "_raw.wav")

                # extract and save spectrogram
                x_raw = inference.file_handler.load(filepath)
                x_spec = inference._preprocess_file(x_raw)

                x_spec_plottable = np.transpose(
                    x_spec[0].cpu().detach().numpy(), (2, 1, 0)
                )
                if i == 0:
                    np.save(out_path_no_extension + "_spec.npy", x_spec_plottable)
                    plt.imshow(x_spec_plottable)
                    plt.savefig(out_path_no_extension + "_spec.png")
                    plt.close()
                    x_spec_norm = (x_spec_plottable - x_spec_plottable.min()) / (
                        x_spec_plottable.max() - x_spec_plottable.min()
                    )

                # create gradCAM explanation
                cam_extractor = GradCAM(inference.model, layer)
                output = inference.model(x_spec)
                cl = output.argmax(dim=1).item()
                activation_map = cam_extractor(cl, output)
                explanation = activation_map[0].numpy().transpose(2, 1, 0)
                np.save(
                    out_path_no_extension + "_exp_" + run_specs + ".npy", explanation
                )

                # create overlay of spectrogram and explanation
                # (the latter has to be interpolated due to varying size)
                heatmap = activation_map[0].squeeze().unsqueeze(0).unsqueeze(0)
                heatmap_resized = F.interpolate(
                    heatmap,  # Add batch and channel dimensions
                    size=x_spec.shape[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                heatmap_resized_norm = (heatmap_resized - heatmap_resized.min()) / (
                    heatmap_resized.max() - heatmap_resized.min()
                )
                heatmap_resized_norm = (
                    heatmap_resized_norm[0].numpy().transpose(2, 1, 0)
                )
                plt.imshow(
                    x_spec_norm, cmap="viridis", aspect="auto"
                )  # Base spectrogram
                plt.imshow(heatmap_resized_norm, cmap="jet", alpha=0.5, aspect="auto")
                plt.savefig(
                    out_path_no_extension + "_highlighted_" + run_specs + ".png"
                )
                plt.close()

                print("saved:", os.path.basename(filename))
