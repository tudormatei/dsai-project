# Full CBM workflow

1. Run `merging_users_together.ipynb` this gives two csv files

   - `merged_dataset_with_concepts.csv` is the file with the actual x,y,z datapoints
   - `merged_windows_labels.csv` is the file with the window definition (window boundaries)

2. Now we want to apply the rule based concepts two these two datasets. Run the `rule_based_labeling.ipynb`.

   - In this file you can change what rule based features / how we calculate them
   - A column in the dataframe is a rule based one for instance `motion_intensity` and `vertical_dominance`
   - This also creates two csvs but now with rules these are the final `datasets` that we can use for modelling

3. Now that we have proper dataset with concepts labeled by us + the rule based calculated ones we can train model to predict the concepts. Run the `sensor_data_to_concepts.ipynb`. This will save the model `sensor_data_to_concept_model.keras`.

4. Now we want to train the second part of the CBM, a light neural network that gets the concepts -> and gives the activity label. Run the `concepts_to_true_labels.ipynb`. This will save the model `concepts_to_true_labels_model.keras`

5. Now to have the full CBM we have to combine the `sensor_data_to_concept_model.keras` then give the output of this to `concepts_to_true_labels_model.keras`. Run the `full_cbm_model.ipynb`.
