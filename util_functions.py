import os
import yaml
import numpy as np


def normalize_audio(data, dtype=np.int16):
    max_val = np.iinfo(dtype).max  # Max possible value for the given dtype
    min_val = np.iinfo(dtype).min  # Min possible value for the given dtype

    # Normalize to [-1, 1]
    data = data / max(abs(data.min()), data.max())

    # Rescale to the dtype range
    data = (data * max_val).astype(dtype)
    return np.clip(data, min_val, max_val)


def get_subdirectories(directory):
    entries = os.listdir(directory)
    subdirectories = [
        entry for entry in entries if os.path.isdir(os.path.join(directory, entry))
    ]
    return subdirectories


def get_medium_values(group, n_samples_per_class):
    mid_point = len(group) // 2
    half_window = n_samples_per_class // 2
    start_idx = max(0, mid_point - half_window)
    end_idx = start_idx + n_samples_per_class
    return group.iloc[start_idx:end_idx]


def get_tasks_from_runspecs(run_spec_list, conf_folder="conf"):
    tasks = []
    task_specs = {}
    task_info = {}
    for run_specs in run_spec_list:
        # we remove the "-16" Specification here
        task = run_specs.split("_")[0]
        short_task = task.split("-")[0]
        tasks.append(short_task)
        if short_task not in task_specs.keys():
            task_specs[short_task] = [run_specs]
        else:
            task_specs[short_task].append(run_specs)

        if short_task not in task_info.keys():
            data_dir = get_yaml_entry_from_file(
                os.path.join(conf_folder, "dataset", task + ".yaml"), "path"
            )
            target_column = get_yaml_entry_from_file(
                os.path.join(conf_folder, "dataset", task + ".yaml"), "target_column"
            )
            index_column = get_yaml_entry_from_file(
                os.path.join(conf_folder, "dataset", task + ".yaml"), "index_column"
            )
            task_info[short_task] = {
                "data_dir": data_dir,
                "target_column": target_column,
                "index_column": index_column,
            }
    tasks = list(set(tasks))
    return tasks, task_specs, task_info


def get_yaml_entry_from_file(yaml_file, entry):
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)

    # Access a specific entry
    entry = data.get(entry)
    return entry


def convert_mel_bins(n_filters):
    # Convert from Mel Scale to Hz scale
    min_freq = 50
    max_freq = 8000
    min_mel = 1127 * np.log1p(min_freq / 700.0)
    max_mel = 1127 * np.log1p(max_freq / 700.0)
    peaks_mel = np.linspace(min_mel, max_mel, n_filters)
    peaks_hz = 700 * (np.expm1(peaks_mel / 1127))
    return peaks_hz


def overlap(existing_labels, new_coords):
    """Check if the new label position overlaps with any previously placed labels."""
    x2_low, x2_high, y2_low, y2_high = new_coords
    for x1_low, x1_high, y1_low, y1_high in existing_labels:
        if (x2_low <= x1_high and x2_high >= x1_low) and (
            y2_low <= y1_high and y2_high >= y1_low
        ):
            return True
    return False


def find_best_label_position(existing_labels, point, text_length, ranges, fontsize):
    """Find the best position for text label close to the point while avoiding overlap and keeping it within plot boundaries."""
    xmax, xmin, ymax, ymin = ranges
    xrange, yrange = xmax - xmin, ymax - ymin

    base_offset = 0.003  # Start with a small offset
    max_attempts = 10  # Limit attempts to prevent infinite loops

    x, y = point
    text_width = fontsize * text_length * 0.0007 * xrange
    text_height = fontsize * 0.002 * yrange

    angles = np.linspace(0, 2 * np.pi, num=16)  # More angles for better placement
    for attempt in range(max_attempts):
        for angle in angles:
            offset = base_offset * (attempt + 1)  # Gradually increase offset if needed
            x_low = x + offset * np.cos(angle)
            y_low = y + offset * np.sin(angle)
            x_high, y_high = x_low + text_width, y_low + text_height

            # Prevent labels from going beyond plot borders
            if x_low < xmin:
                x_low, x_high = xmin, xmin + text_width
            if x_high > xmax:
                x_low = xmax - text_width * 0.1  # Fix: Keep it closer to original point
                x_high = xmax
            if y_low < ymin:
                y_low, y_high = ymin, ymin + text_height
            if y_high > ymax:
                y_high, y_low = ymax, ymax - text_height

            new_coords = (x_low, x_high, y_low, y_high)
            if not overlap(existing_labels, new_coords):
                # Adjust text alignment dynamically based on proximity to edges
                ha = "center"
                if x_low <= xmin + 0.02 * xrange:  # Close to left border
                    ha = "left"
                elif x_high >= xmax - 0.02 * xrange:  # Close to right border
                    ha = "right"
                return new_coords, ha

    return (
        new_coords,
        "center",
    )  # Return last tried position with default alignment as fallback


SCORING_FUNCTION = "CumulativeAccuracy"
ROOT_ANALYSIS_DIR = "results/analysis"
ROOT_RESULT_DIR = get_yaml_entry_from_file("conf/config.yaml", "results_dir")
EXPERIMENT_ID = get_yaml_entry_from_file("conf/config.yaml", "experiment_id")
ROOT_EXPERIMENT_DIR = os.path.join(ROOT_RESULT_DIR, EXPERIMENT_ID)


ROOT_TRAINING_DIR = os.path.join(ROOT_EXPERIMENT_DIR, "training")
os.makedirs(ROOT_ANALYSIS_DIR, exist_ok=True)

SAMPLE_DIFFICULTY_DIR = os.path.join(
    ROOT_EXPERIMENT_DIR, "curriculum/CumulativeAccuracy"
)
RUN_SPEC_LIST = get_subdirectories(ROOT_TRAINING_DIR)

TASKS, TASK_SPECS, TASK_INFO = get_tasks_from_runspecs(RUN_SPEC_LIST)
