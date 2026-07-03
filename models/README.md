# Models Directory

This directory must contain one subfolder per deployed model used by the dashboard application.

## Folder Naming
The subfolder name is the unique model identifier (e.g., `shivering_hen`).

## Required Files (inside each model folder)
- `model.xgb` : The serialized XGBoost model file.
- `config.yaml` : File that contains the XGBoost model configuration.
- `test_dataset.csv` : Test dataset that the XGBoost model was tested.

## Example Structure
```
models/
    shivering_hen/
        model.xgb
        config.yaml
        test_dataset.csv
    crawling_wren/
        model.xgb
        config.yaml
        test_dataset.csv
```