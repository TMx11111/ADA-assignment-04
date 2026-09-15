# %% [markdown]
# # Assignment 4: Controls and prediction
# This notebook extends Assignment 3 using Week 8 partialling out and cross-fitting.
# Run all cells in order. Local competition data are preferred; the Kaggle API is
# used only if the data are missing. Outputs go to `Assignment04/output/`.
# The final report body is limited to 300 words; the notebook documents the analysis.

# %%
from pathlib import Path
import os
import json
import hashlib
import platform
import zipfile
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sklearn
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import KFold
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from IPython.display import display, Image, Markdown

# One BLAS thread avoids overhead for these relatively small regressions.
thread_limit = threadpool_limits(limits=1)
warnings.filterwarnings('error', category=ConvergenceWarning)
COMPETITION_ID = 'ecom-90025-2026-sm-2-ada-assignment-and-practice'
origin = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
search_roots = [origin, *origin.parents]
ADA_DIR = next((p for p in search_roots if (p / 'Assignment04').is_dir()), None)
if ADA_DIR is None:
    ADA_DIR = next((p / 'ADA' for p in search_roots
                    if (p / 'ADA' / 'Assignment04').is_dir()), origin)
WORK_DIR = ADA_DIR / 'Assignment04' if (ADA_DIR / 'Assignment04').is_dir() else origin
OUT_DIR = WORK_DIR / 'output'
OUT_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR = Path(os.environ.get('ADA_DATA_DIR', str(ADA_DIR / COMPETITION_ID)))

# Decisions fixed before validation: match A3's splits, grid and top-12 rule.
SEED = 1
N_OUTER, N_CROSSFIT = 5, 3
ALPHAS = np.logspace(0.5, -2.5, 60)
TOP_K = 12
FEATURES = [f'X{i}' for i in range(1, 51)]
FOCAL = 'X34'  # fixed from Assignment 1, not selected using A4 validation outcomes
CONTROLS = [c for c in FEATURES if c != FOCAL]
print('Python:', platform.python_version(), '| sklearn:', sklearn.__version__)
print('Data:', DATA_DIR)

# %% [markdown]
# ## 1. Data: the Assignment 1 import pattern
# Read `train_data.csv` and `test_data.csv` with pandas and export `ID,Y` without
# a pandas index. For online replication, join the competition and configure your
# own Kaggle credentials (`KAGGLE_API_TOKEN` or the standard Kaggle configuration).
# No token is stored in this notebook. `ID` is an identifier, not a regressor.

# %%
required_files = ['train_data.csv', 'test_data.csv', 'submission.csv']
if not all((DATA_DIR / name).is_file() for name in required_files):
    from kaggle.api.kaggle_api_extended import KaggleApi
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    api = KaggleApi()
    api.authenticate()
    api.competition_download_files(competition=COMPETITION_ID, path=str(DATA_DIR))
    with zipfile.ZipFile(DATA_DIR / f'{COMPETITION_ID}.zip') as archive:
        # Extract only the three known CSV members.
        for name in required_files:
            member = next(s for s in archive.namelist() if Path(s).name == name)
            (DATA_DIR / name).write_bytes(archive.read(member))

train_df = pd.read_csv(DATA_DIR / 'train_data.csv')
test_df = pd.read_csv(DATA_DIR / 'test_data.csv')
template = pd.read_csv(DATA_DIR / 'submission.csv')
assert set(train_df.columns) == set(FEATURES + ['ID', 'Y'])
assert set(test_df.columns) == set(FEATURES + ['ID'])
assert list(template.columns) == ['ID', 'Y']
assert train_df.ID.is_unique and test_df.ID.is_unique
assert set(train_df.ID).isdisjoint(set(test_df.ID))
assert template.ID.equals(test_df.ID), 'Template and test IDs must match in order.'
assert np.isfinite(train_df[FEATURES + ['Y']].to_numpy()).all()
assert np.isfinite(test_df[FEATURES].to_numpy()).all()
X, y = train_df[FEATURES].to_numpy(), train_df.Y.to_numpy()
X_test = test_df[FEATURES].to_numpy()
d_index = FEATURES.index(FOCAL)
w_indices = [FEATURES.index(c) for c in CONTROLS]
data_audit = pd.DataFrame({
    'sample': ['training', 'test'], 'rows': [len(train_df), len(test_df)],
    'predictors': [len(FEATURES)] * 2,
    'missing_cells': [int(train_df.isna().sum().sum()), int(test_df.isna().sum().sum())]
})
display(data_audit)
display(train_df[['ID', 'Y', FOCAL, 'X21', 'X45']].head())

# %% [markdown]
# ## 2. What do controls change?
# Set $D=X_{34}$ and $W$ to the remaining 49 variables. Compare the marginal OLS
# slope with the slope controlling for $W$. The Frisch–Waugh–Lovell identity gives
# the same controlled slope by regressing residualised $Y$ on residualised $D$.
# Neither coefficient is a causal effect: variable meanings, treatment assignment,
# and conditional ignorability are unknown. The FWL calculation is descriptive.

# %%
D, W = X[:, d_index], X[:, w_indices]
naive_ols = LinearRegression().fit(D[:, None], y)
controlled_ols = LinearRegression().fit(np.column_stack([D, W]), y)
residual_y = y - LinearRegression().fit(W, y).predict(W)
residual_d = D - LinearRegression().fit(W, D).predict(W)
fwl_slope = float(residual_d @ residual_y / (residual_d @ residual_d))
assert np.isclose(fwl_slope, controlled_ols.coef_[0], atol=1e-10)
association_table = pd.DataFrame({
    'specification': ['X34 alone', 'X34 + 49 linear controls', 'FWL residual regression'],
    'X34_slope': [naive_ols.coef_[0], controlled_ols.coef_[0], fwl_slope]
})
display(association_table.round(6))
association_table.to_csv(OUT_DIR / 'association_coefficients.csv', index=False)

# %% [markdown]
# ## 3. Reproduce the degree-3 baseline
# Follow Assignment 3: fit degree-2 LASSO, rank original variables by absolute
# coefficient mass, add degree-3 products among the top 12, then refit LASSO.
# Scaling, ranking and AICc penalty choice are repeated inside each training split.
# AICc counts nonzero coefficients plus the intercept, matching the supplied A3
# script. It is a model-selection heuristic, not an exact degrees-of-freedom result.

# %%
def mse(actual, predicted):
    return float(np.mean((np.asarray(actual) - predicted) ** 2))


def fit_lasso_aicc(matrix, target):
    scaler = StandardScaler().fit(matrix)
    scaled = np.asfortranarray(scaler.transform(matrix))
    walker = Lasso(warm_start=True, max_iter=20000, tol=1e-4)
    best_value, best_alpha = np.inf, None
    for alpha in ALPHAS:
        walker.set_params(alpha=float(alpha))
        walker.fit(scaled, target)
        k = int(np.count_nonzero(walker.coef_)) + 1
        error = mse(target, walker.predict(scaled))
        if k >= len(target) - 2 or error <= 0:
            break
        aicc = len(target) * np.log(error) + 2*k + 2*k*(k+1)/(len(target)-k-1)
        if aicc < best_value:
            best_value, best_alpha = aicc, float(alpha)
    if best_alpha is None:
        raise RuntimeError('No finite AICc model on the pre-specified grid.')
    model = Lasso(alpha=best_alpha, max_iter=20000, tol=1e-4).fit(scaled, target)
    return {'scaler': scaler, 'model': model, 'alpha': best_alpha,
            'nonzero': int(np.count_nonzero(model.coef_))}


def predict_lasso(fit, matrix):
    return fit['model'].predict(fit['scaler'].transform(matrix))


def fit_degree3(matrix, target):
    poly2 = PolynomialFeatures(2, include_bias=False)
    quadratic = poly2.fit_transform(matrix)
    fit2 = fit_lasso_aicc(quadratic, target)
    mass = (poly2.powers_ * np.abs(fit2['model'].coef_)[:, None]).sum(axis=0)
    selected = np.argsort(mass)[::-1][:min(TOP_K, matrix.shape[1])]
    poly3 = PolynomialFeatures(3, include_bias=False).fit(matrix[:, selected])
    cubic_mask = poly3.powers_.sum(axis=1) == 3
    cubic = poly3.transform(matrix[:, selected])[:, cubic_mask]
    fit3 = fit_lasso_aicc(np.column_stack([quadratic, cubic]), target)
    return {'poly2': poly2, 'poly3': poly3, 'cubic_mask': cubic_mask,
            'selected': selected, 'fit2': fit2, 'fit3': fit3}


def predict_degree3(fit, matrix):
    quadratic = fit['poly2'].transform(matrix)
    cubic = fit['poly3'].transform(matrix[:, fit['selected']])[:, fit['cubic_mask']]
    return predict_lasso(fit['fit3'], np.column_stack([quadratic, cubic]))

# %% [markdown]
# ## 4. New ideas: cross-fitted control adjustment
# Learn $l(W)=E[Y\mid W]$ with the selected-cubic LASSO and
# $m_j(W)=E[D^j\mid W]$ with quadratic LASSO, for $j=1,2,3$.
# Three-fold cross-fitting creates held-out residuals $U=Y-\hat l(W)$ and
# $V_j=D^j-\hat m_j(W)$ inside each outer training fold. OLS on these residuals fits:
#
# 1. **Constant slope:** $U=\theta V_1+e$.
# 2. **Varying slope:** $U=V_1(\theta+W'\delta)+e$.
# 3. **Varying slope + powers:** also include $V_2$ and $V_3$.
#
# Controls in the varying slope are centred/scaled using the outer training sample.
# Because $W$ is held fixed, residualising $D W_k$ gives $W_k V_1$ exactly.
# The third specification combines Week 8 partialling out with Week 3 products;
# it is an explicit extension, not the exact `LinearDML` lecture example.
# Refit nuisance models on the outer training data and predict
# $\hat Y=\hat l(W)+\hat\theta'(B-\widehat{E[B\mid W]})$.
# Omitting the nuisance component would predict residuals, not the outcome.
# Cross-fitting protects residual construction; it does not guarantee better MSE.

# %%
CONTROL_MODELS = ['CF constant slope', 'CF varying slope', 'CF varying slope + powers']


def fit_nuisances(w, d, target):
    outcome = fit_degree3(w, target)
    poly = PolynomialFeatures(2, include_bias=False)
    z = poly.fit_transform(w)
    treatments = [fit_lasso_aicc(z, d**power) for power in (1, 2, 3)]
    return {'outcome': outcome, 'poly': poly, 'treatments': treatments}


def nuisance_predictions(fit, w):
    z = fit['poly'].transform(w)
    return (predict_degree3(fit['outcome'], w),
            np.column_stack([predict_lasso(m, z) for m in fit['treatments']]))


def residual_designs(d, conditional_powers, scaled_w):
    residuals = np.column_stack([d**power for power in (1, 2, 3)]) - conditional_powers
    v = residuals[:, :1]
    varying = np.column_stack([v, v * scaled_w])
    return [v, varying, np.column_stack([varying, residuals[:, 1:]])]


def fit_control_models(matrix, target, seed):
    d, w = matrix[:, d_index], matrix[:, w_indices]
    l_oof = np.full(len(target), np.nan)
    m_oof = np.full((len(target), 3), np.nan)
    covered = np.zeros(len(target), dtype=int)
    splits = KFold(N_CROSSFIT, shuffle=True, random_state=seed)
    for inner, (train, valid) in enumerate(splits.split(w), 1):
        nuisances = fit_nuisances(w[train], d[train], target[train])
        l_oof[valid], m_oof[valid] = nuisance_predictions(nuisances, w[valid])
        covered[valid] += 1
        print(f'    nuisance cross-fit {inner}/{N_CROSSFIT}', flush=True)
    assert (covered == 1).all() and np.isfinite(m_oof).all()
    scaler = StandardScaler().fit(w)
    designs = residual_designs(d, m_oof, scaler.transform(w))
    regressions = [LinearRegression(fit_intercept=False).fit(z, target-l_oof)
                   for z in designs]
    assert all(np.linalg.matrix_rank(z) == z.shape[1] for z in designs)
    final_nuisances = fit_nuisances(w, d, target)
    return {'scaler': scaler, 'regressions': regressions, 'nuisances': final_nuisances,
            'residual_d_variance': float(np.mean((d-m_oof[:, 0])**2)),
            'theta_constant': float(regressions[0].coef_[0])}


def predict_controls(fit, matrix):
    d, w = matrix[:, d_index], matrix[:, w_indices]
    l_hat, m_hat = nuisance_predictions(fit['nuisances'], w)
    designs = residual_designs(d, m_hat, fit['scaler'].transform(w))
    return {name: l_hat + model.predict(z)
            for name, model, z in zip(CONTROL_MODELS, fit['regressions'], designs)}

# %% [markdown]
# ## 5. Common outer validation
# All seven candidates use the same five shuffled folds as Assignment 3 (seed 1).
# Outer validation outcomes never enter fitting, scaling, AICc, variable ranking,
# or nuisance cross-fitting. Select the candidate with the lowest pooled OOF MSE;
# exact ties prefer the earlier column. This compares a fixed candidate set, not a
# separately nested estimate of the final model-selection procedure. Reusing this
# dataset across assignments also limits claims about generalisation.
# A3's reported stack (6.023 CV; 5.70982 public MSE) is historical context only;
# its seven-model ensemble is not reconstructed from the inconsistent old CSVs.

# %%
MODEL_NAMES = ['OLS X34', 'OLS all 50', 'Degree-2 LASSO', 'A3 degree-3 LASSO'] + CONTROL_MODELS
oof = pd.DataFrame(np.nan, index=train_df.index, columns=MODEL_NAMES)
fold_id = np.full(len(y), -1)
fold_details = []
for fold, (train, valid) in enumerate(KFold(N_OUTER, shuffle=True, random_state=SEED).split(X), 1):
    print(f'Outer fold {fold}/{N_OUTER}', flush=True)
    fold_id[valid] = fold
    univariate = LinearRegression().fit(X[train, d_index, None], y[train])
    oof.loc[valid, 'OLS X34'] = univariate.predict(X[valid, d_index, None])
    linear = LinearRegression().fit(X[train], y[train])
    oof.loc[valid, 'OLS all 50'] = linear.predict(X[valid])
    baseline = fit_degree3(X[train], y[train])
    oof.loc[valid, 'Degree-2 LASSO'] = predict_lasso(baseline['fit2'], baseline['poly2'].transform(X[valid]))
    oof.loc[valid, 'A3 degree-3 LASSO'] = predict_degree3(baseline, X[valid])
    controls_fit = fit_control_models(X[train], y[train], SEED + 100 + fold)
    for name, predictions in predict_controls(controls_fit, X[valid]).items():
        oof.loc[valid, name] = predictions
    fold_details.append({'fold': fold, 'degree3_alpha': baseline['fit3']['alpha'],
                         'degree3_nonzero': baseline['fit3']['nonzero'],
                         'degree3_variables': ', '.join(FEATURES[j] for j in baseline['selected']),
                         'constant_slope': controls_fit['theta_constant'],
                         'residual_D_variance': controls_fit['residual_d_variance']})
    print(oof.loc[valid].apply(lambda p: mse(y[valid], p)).round(4).to_string(), flush=True)

assert np.isfinite(oof.to_numpy()).all() and (fold_id > 0).all()
fold_scores = pd.DataFrame([
    {'model': name, 'fold': fold, 'MSE': mse(y[fold_id == fold], oof.loc[fold_id == fold, name])}
    for name in MODEL_NAMES for fold in range(1, N_OUTER+1)
])
comparison = pd.DataFrame({
    'model': MODEL_NAMES,
    'OOF_MSE': [mse(y, oof[name].to_numpy()) for name in MODEL_NAMES],
    'OOF_R2': [1 - mse(y, oof[name].to_numpy()) / np.var(y) for name in MODEL_NAMES]
}).sort_values('OOF_MSE', kind='stable').reset_index(drop=True)
winner = comparison.loc[0, 'model']
display(comparison.round(6))
display(pd.DataFrame(fold_details))
comparison.to_csv(OUT_DIR / 'model_comparison.csv', index=False)
fold_scores.to_csv(OUT_DIR / 'fold_mse.csv', index=False)
pd.DataFrame(fold_details).to_csv(OUT_DIR / 'fold_details.csv', index=False)
oof.assign(ID=train_df.ID, Y=y, fold=fold_id).to_csv(OUT_DIR / 'oof_predictions.csv', index=False)

# %% [markdown]
# ## 6. Diagnostics and interpretation
# Paired fold differences compare each control model with the refitted degree-3
# baseline. Positive differences favour the control model. Five folds share training
# observations, so their dispersion is descriptive and is not an independent-sample
# confidence interval. Prediction accuracy and coefficient interpretation are
# separate objectives; an unsuccessful control model remains a useful finding.

# %%
fold_pivot = fold_scores.pivot(index='fold', columns='model', values='MSE')
paired = pd.DataFrame({name: fold_pivot['A3 degree-3 LASSO']-fold_pivot[name]
                       for name in CONTROL_MODELS})
display(paired.round(6))
paired.to_csv(OUT_DIR / 'paired_fold_improvements.csv')
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
ordered = comparison.iloc[::-1]
axes[0].barh(ordered.model, ordered.OOF_MSE, color='#4472a5')
axes[0].set_xlabel('Out-of-fold MSE (lower is better)')
axes[0].set_title('Same five outer folds')
for name in CONTROL_MODELS:
    axes[1].plot(paired.index, paired[name], marker='o', label=name.replace('CF ', ''))
axes[1].axhline(0, color='black', linewidth=0.8)
axes[1].set_xticks(range(1, N_OUTER+1))
axes[1].set_xlabel('Outer fold')
axes[1].set_ylabel('Baseline MSE minus control-model MSE')
axes[1].set_title('Gain relative to A3 degree-3 LASSO')
axes[1].legend(fontsize=8)
fig.tight_layout()
fig.savefig(OUT_DIR / 'validation_comparison.png', dpi=180, bbox_inches='tight')
fig.savefig(OUT_DIR / 'validation_comparison.pdf', bbox_inches='tight')
plt.close(fig)
display(Image(filename=str(OUT_DIR / 'validation_comparison.png')))

# %% [markdown]
# ## 7. Full-data fit and Kaggle file
# Refit the chosen procedure on all training rows. Test outcomes are unavailable;
# no public leaderboard score is inferred from cross-validation. The user's chosen
# workflow is manual Kaggle submission and screenshot capture.

# %%
print('Refitting selected model:', winner, flush=True)
if winner in CONTROL_MODELS:
    final_fit = fit_control_models(X, y, SEED + 999)
    test_predictions = predict_controls(final_fit, X_test)[winner]
elif winner in ['A3 degree-3 LASSO', 'Degree-2 LASSO']:
    final_fit = fit_degree3(X, y)
    test_predictions = (predict_degree3(final_fit, X_test) if winner == 'A3 degree-3 LASSO'
                        else predict_lasso(final_fit['fit2'], final_fit['poly2'].transform(X_test)))
else:
    cols = [d_index] if winner == 'OLS X34' else list(range(len(FEATURES)))
    final_fit = LinearRegression().fit(X[:, cols], y)
    test_predictions = final_fit.predict(X_test[:, cols])

submission = pd.DataFrame({'ID': test_df.ID, 'Y': test_predictions})
assert list(submission.columns) == list(template.columns)
assert submission.ID.equals(template.ID)
assert len(submission) == len(test_df) and np.isfinite(submission.Y).all()
submission_path = OUT_DIR / 'submission.csv'
submission.to_csv(submission_path, index=False)
roundtrip = pd.read_csv(submission_path)
assert roundtrip.ID.equals(test_df.ID)
assert np.allclose(roundtrip.Y, test_predictions, rtol=1e-12, atol=1e-12)
display(submission.head())
print(f'Saved {len(submission):,} predictions to {submission_path}')

# %% [markdown]
# ## 8. Report body, evidence and remaining submission steps
# The report below is generated from the actual run. Tables, figure captions and
# the separate AI declaration are outside the report body. Add your real Kaggle
# score screenshot, circle the group name, and supply a public link to
# `assignment04.py` before assembling the final PDF. The local code link does not
# satisfy the public-link requirement. Group membership should be checked against A3.

# %%
scores = comparison.set_index('model').OOF_MSE
best_control = min(CONTROL_MODELS, key=lambda name: scores[name])
gain = scores['A3 degree-3 LASSO'] - scores[best_control]
direction = 'reduced' if gain > 0 else 'increased'
historical_relation = 'lower' if 6.023 < scores[winner] else 'no lower'
report_body = f'''Assignment 3 reported cross-validated MSE 6.023 for its stack and public MSE 5.70982. We extend its selected-cubic LASSO with Week 8 control adjustment, retaining X34 as the focal variable from Assignment 1.

Controlling for the other 49 variables changes the linear X34 slope from {naive_ols.coef_[0]:.3f} to {fwl_slope:.3f}. Residualising both outcome and predictor reproduces the controlled OLS coefficient. Because the variables are anonymous and treatment assignment is unknown, these are conditional associations, not identified causal effects.

We cross-fit LASSO nuisance regressions for the outcome and powers of X34. Residual regressions allow a constant slope, a slope varying linearly with controls, or a varying slope plus quadratic and cubic X34 terms. Nuisance fitting, scaling, variable selection and penalty choice exclude the observations being scored. Predictions restore the nuisance outcome component.

On identical five-fold splits, the reproduced degree-3 baseline scores {scores['A3 degree-3 LASSO']:.3f}. The best control specification, {best_control.replace('CF ', '')}, scores {scores[best_control]:.3f}; it {direction} MSE by {abs(gain):.3f}. {winner} has the lowest MSE among the seven candidates ({scores[winner]:.3f}) and supplies the {len(test_df):,} test predictions after full-sample refitting. These validation scores compare fixed candidates; selecting the winner can introduce optimism. The gain is small relative to fold variation. The earlier stack's reported MSE is {historical_relation}, although that ensemble was not rerun.

Kaggle submission and the score screenshot are pending; no new public score is claimed.'''
word_count = len(report_body.split())
assert word_count <= 300, f'Report body has {word_count} words.'
display(Markdown(report_body))
print('Report body word count:', word_count)

ai_declaration = '''## AI Use Declaration

OpenAI Codex assisted with reading the supplied report and lecture notebooks,
designing and implementing the control specifications, running validation,
producing figures and predictions, and drafting this report. The report and
interpretations require the authors' review. No other AI assistance is claimed
by this workflow.

Material user prompt (original):
“参考 [ADA](ADA/) 文件夹中的相关资料，基于 [Assignment3_Report-fin.pdf](ADA/Assignment03/Assignment3_Report-fin.pdf) (用 [pdf-parse.cmd](pdf-parse.cmd) 看pdf文件) , 完成 [Assignment04 - 副本.ipynb](ADA/Assignment04/Assignment04 - 副本.ipynb) , 导入数据的代码参考 [Assignment01_solution.ipynb](ADA/Assignment01 sol/Assignment01_solution.ipynb)， 数据集在 [ecom-90025-2026-sm-2-ada-assignment-and-practice](ADA/ecom-90025-2026-sm-2-ada-assignment-and-practice/)， [$grill-me](C:/Users/12768/.codex/skills/grill-me/SKILL.md)有需要进一步说明的问我”

Codex asked whether to stop at the local notebook, predictions and report body or
also submit to Kaggle and complete the PDF. The user selected:
“先完成这些本地文件，Kaggle 提交和截图由我处理”.

The user requested the data sources and local deliverables. Codex proposed the
specific control models and validation implementation; these modelling choices
must not be represented as independently devised or already endorsed by the authors.
'''
table_lines = ['| Model | OOF MSE | OOF R2 |', '|---|---:|---:|']
for row in comparison.itertuples(index=False):
    table_lines.append(f'| {row.model} | {row.OOF_MSE:.6f} | {row.OOF_R2:.6f} |')
report = ('# Assignment 4: Controls and prediction\n\n'
          'Draft: add group cover, real Kaggle evidence and public code URL before LMS submission.\n\n'
          + report_body + f'\n\n*Report body: {word_count} words.*\n\n'
          + '\n'.join(table_lines)
          + '\n\nTable 1. Identical five-fold validation, seed 1. Lower MSE is better. '
          'The A3 stack is not included in this new comparison.\n\n'
          + '![Validation comparison](validation_comparison.png)\n\n'
          + 'Figure 1. Candidate MSE and paired fold gains relative to the refitted degree-3 baseline.\n\n'
          + '## Code and sources\n\nPublic Python code URL: **pending author publication**.\n\n'
          + f'Data: https://www.kaggle.com/competitions/{COMPETITION_ID}/data\n\n'
          + 'Course sources: Week03_regression.ipynb (interactions and cross-validation); '
          + 'Week04_LASSO.ipynb (scaling, LASSO and AICc); Week08_controls.ipynb '
          + '(FWL, cross-fitting and heterogeneous slopes); Assignment3_Report-fin.pdf '
          + '(historical results); supplied Assignment 3 Python script (baseline implementation).\n\n'
          + ai_declaration)
(OUT_DIR / 'Assignment4_Report_draft.md').write_text(report, encoding='utf-8')
manifest = {
    'seed': SEED, 'outer_folds': N_OUTER, 'nuisance_crossfit_folds': N_CROSSFIT,
    'focal_variable': FOCAL, 'top_k': TOP_K, 'alphas': ALPHAS.tolist(),
    'selected_model': winner, 'oof_mse': float(scores[winner]),
    'report_body_words': word_count, 'kaggle_submitted': False,
    'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                 'pandas': pd.__version__, 'sklearn': sklearn.__version__,
                 'matplotlib': matplotlib.__version__},
    'input_sha256': {name: hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest()
                     for name in required_files},
    'submission_sha256': hashlib.sha256(submission_path.read_bytes()).hexdigest()
}
(OUT_DIR / 'run_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
print('Report:', OUT_DIR / 'Assignment4_Report_draft.md')

# %%
# After manual submission, save the genuine screenshot at the following path and
# rerun this cell to display it in the notebook. Use the same image in the final PDF.
screenshot_path = OUT_DIR / 'kaggle_score.png'
if screenshot_path.is_file():
    display(Image(filename=str(screenshot_path)))
else:
    print('Kaggle score screenshot pending: submit output/submission.csv, then save '
          'the genuine screenshot as output/kaggle_score.png.')
