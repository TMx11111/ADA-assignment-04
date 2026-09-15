# Assignment 4

Open **Assignment04 - .ipynb** and run its cells in order, or run the equivalent
single Python file from the workspace root:

```powershell
.\.venv\Scripts\python.exe ADA\Assignment04\assignment04.py
```

For another Python environment, install the dependencies first:

```text
python -m pip install numpy pandas scikit-learn matplotlib ipython threadpoolctl kaggle
```

The notebook additionally needs a Python Jupyter kernel. Exact versions used for
the completed run are recorded in `output/run_manifest.json`. The script and
notebook contain the same analysis; either can reproduce the result independently.
The unchanged `Assignment04.ipynb` is the original assignment brief.

Data are read from `../ecom-90025-2026-sm-2-ada-assignment-and-practice/` relative to
this folder. Set `ADA_DATA_DIR` to use another location. If the three required CSVs
are absent, the code downloads them through the Kaggle API, following Assignment 1.
The person running the code must join the competition and configure their own Kaggle
credentials. The Python file can also be published and run by itself; it then uses
its containing directory as its working directory. No credentials are embedded.

## Analysis

- Reproduce the Assignment 3 selected-cubic LASSO baseline using its five outer
  folds, penalty grid, AICc convention and training-only top-12 selection.
- Demonstrate Week 8 FWL control adjustment for X34, fixed from Assignment 1.
- Compare constant and varying slopes estimated using three-fold cross-fitted
  residuals, including an extension with quadratic and cubic X34 terms.
- Score seven fixed candidates on the same five outer folds and refit the lowest
  MSE candidate on all training observations.

The cross-fitting is implemented directly in Python so the residual construction
and restoration of the conditional outcome component are visible. The varying
slope plus powers specification composes Week 8 methods with the products used
in Week 3; it is not represented as an unchanged lecture example. These anonymous
variables support an association/prediction analysis, not a causal claim.

Assignment 3's final PDF is the source of historical stack scores. Existing CSVs
in Assignment03/output describe a different, earlier model set and are not loaded.
The current notebook does not rerun the old seven-member stack or assume its test
predictions are the new model's predictions.

## Deliverables

- `output/submission.csv`: 1,600 predictions in the original test ID order.
- `output/Assignment4_Report_draft.md`: report body (at most 300 words), results
  table, figure, source information and a separate AI Use Declaration.
- `output/model_comparison.csv`, `fold_mse.csv`, `fold_details.csv`,
  `oof_predictions.csv`, `paired_fold_improvements.csv`: reproducible validation evidence.
- `output/association_coefficients.csv`: marginal, controlled and FWL coefficients.
- `output/validation_comparison.png` and `.pdf`: exported analysis figure.
- `output/run_manifest.json`: configuration, versions and input/output SHA-256 hashes.

As requested, Kaggle submission and the genuine score screenshot remain with the
user. Submit `output/submission.csv`, circle the group name in the score screenshot,
and save it as `output/kaggle_score.png` to display it in the final notebook cell.
Before creating the LMS PDF, add the current group cover page, the genuine Kaggle
evidence and a public URL to **assignment04.py**. A local file link does not satisfy
the assignment's public-code requirement. Do not paste Python code into the PDF.
Review the AI declaration and disclose any additional assistance used afterwards.
