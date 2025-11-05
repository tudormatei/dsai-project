# XAI for TS Group 1 Project

This folder contains a self-contained, step-by-step implementation of a Concept Bottleneck Model (CBM) pipeline for sensor-based activity recognition. The README explains the purpose of each top-level file and subfolder, the expected inputs and outputs for each notebook or script, assumptions, and quick-start notes so collaborators can reproduce results or continue development.

## High-level overview

Purpose: the pipeline converts raw sensor windows into interpretable concept labels (rule-based + learned), then uses those concepts to predict activity labels. It is organized so each notebook performs one logical step and writes intermediate CSVs or model files that the next step consumes.

Core notebooks (run in this order):

- `1_merging_users_together.ipynb` — Merge per-user windowed sensor CSVs into unified datasets. Produces `data/merged_dataset_with_concepts.csv` (sensor windows with ground-truth concept labels where available) and `data/merged_window_labels.csv` (window boundary metadata).
- `2_rule_based_labeling.ipynb` — Compute deterministic, rule-based concept features (e.g., motion intensity, vertical dominance). Takes merged CSVs and produces augmented CSVs with rule-based concept columns.
- `3_sensor_data_to_concepts.ipynb` — Train a model that maps sensor windows (x,y,z) to concept probabilities/labels. Saves `models/sensor_data_to_concept_model.keras`.
- `4_concept_to_true_labels.ipynb` — Train a lightweight model that maps concept vectors to activity labels. Saves `models/concepts_to_true_labels_model.keras`.
- `5_models_stitched_together.ipynb` — Stitch the two trained models into an end-to-end pipeline: sensor -> predicted concepts -> predicted activity. Useful for inference and error analysis.
- `CBM.ipynb` — End-to-end training and evaluation harness that trains both parts (concept and label predictors) using the same splits and produces final metrics and visualizations.

## Directories and their contents

- `benchmarks/`

  - `benchmarks.md` — Notes and results from benchmarking experiments and comparisons between models or design choices.

- `data/` — Central storage for datasets produced and used by the notebooks. Files here are read/written by the pipeline notebooks.

  - `dataset_with_concepts_user{N}.csv` — Per-user windowed datasets including sensor columns and (optionally) manually labeled concepts. Useful for per-user analysis or when re-merging subsets.
  - `final_dataset.csv` — Final cleaned dataset produced after merging and applying rule-based concepts; ready for training the models.
  - `final_window_labels.csv` — Final window boundary metadata matching `final_dataset.csv`.
  - `merged_dataset_with_concepts.csv` — Output of `1_merging_users_together.ipynb`, contains merged sensor windows and concept columns.
  - `merged_window_labels.csv` — Output of `1_merging_users_together.ipynb`, the canonical window definitions for the merged dataset.
  - `window_labels_user{N}.csv` — Per-user window metadata before merging.
  - `combinations/` — Precomputed combinations of iterations or user subsets used in some experiments; useful for replication and ablation studies.
  - `iteration{1,2,3}/` — Saved intermediate results for each labeling iteration. Each iteration folder mirrors the per-user datasets and merged outputs for reproducibility.

- `models/`

  - `sensor_data_to_concept_model.keras` — Trained model mapping sensor windows to concept labels.
  - (other model files may be stored here after training)

- `utils/` — Helper notebooks and utilities used during development and debugging.
  - `blackbox_model.ipynb` — Experiments with un-interpretable baseline models used for comparison to CBM.
  - `concept_labeling_reliablity_checker.ipynb` — Tools to compute inter-rater reliability and sanity checks for concept labels.
  - `concept_labeling.ipynb` — Interactive tools used to collect or inspect human concept labels.
  - `dataset_visualization.ipynb` — Visualizations used to inspect the datasets, concept distributions, and sensor signals.

## Full CBM workflow

1. Run `1_merging_users_together.ipynb` this gives two csv files

   - `merged_dataset_with_concepts.csv` is the file with the actual x,y,z datapoints
   - `merged_windows_labels.csv` is the file with the window definition (window boundaries)

2. Now we want to apply the rule based concepts two these two datasets. Run the `2_rule_based_labeling.ipynb`.

   - In this file you can change what rule based features / how we calculate them
   - A column in the dataframe is a rule based one for instance `motion_intensity` and `vertical_dominance`
   - This also creates two csvs but now with rules these are the final `datasets` that we can use for modelling

3. Now that we have proper dataset with concepts labeled by us + the rule based calculated ones we can train model to predict the concepts. Run the `3_sensor_data_to_concepts.ipynb`. This will save the model `sensor_data_to_concept_model.keras`.

4. Now we want to train the second part of the CBM, a light neural network that gets the concepts -> and gives the activity label. Run the `4_concepts_to_true_labels.ipynb`. This will save the model `concepts_to_true_labels_model.keras`

5. Now to have the full CBM we have to combine the `sensor_data_to_concept_model.keras` then give the output of this to `concepts_to_true_labels_model.keras`. Run the `5_models_stitched_together.ipynb`.

6. `CBM.ipynb` is the file that puts everything together, training and testing both the concept and activity predictors on the same training and test set. This is the proper end-to-end CBM architecture.
