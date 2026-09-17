#!/usr/bin/env python3
"""Assignment 4: controls and single-model prediction (Weeks 3, 4 and 8).

Run: python assignment04.py --data-dir ../data
Use --check-kaggle to check access, or --submit-to-kaggle to upload saved predictions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from threadpoolctl import threadpool_limits


HERE = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
COMPETITION = 'ecom-90025-2026-sm-2-ada-assignment-and-practice'
FEATURES = [f'X{i}' for i in range(1, 51)]
FOCAL = 'X34'
D_INDEX = FEATURES.index(FOCAL)
W_INDICES = [i for i, name in enumerate(FEATURES) if name != FOCAL]
SEED = 1
OUTER_FOLDS = 5
CROSSFIT_FOLDS = 4
# Inherit A3's fixed grid for its degree-2 baseline; AICc is taught in Week 4.
# This grid is fixed before looking at the A4 validation results.
ALPHAS = np.logspace(0.5, -2.5, 60)
MODEL_NAMES = [
    'OLS: X34 alone',
    'OLS: all 50 variables',
    'Quadratic LASSO',
    'Cross-fitted constant slope',
]


def mse(y, prediction):
    return float(np.mean((np.asarray(y) - np.asarray(prediction)) ** 2))


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def get_data(data_dir):
    """Week 1 KaggleApi authentication; credentials stay outside this code."""
    names = ['train_data.csv', 'test_data.csv']
    if not all((data_dir / name).is_file() for name in names):
        from kaggle.api.kaggle_api_extended import KaggleApi

        data_dir.mkdir(parents=True, exist_ok=True)
        api = KaggleApi()
        api.authenticate()
        api.competition_download_files(competition=COMPETITION, path=str(data_dir))
        with zipfile.ZipFile(data_dir / f'{COMPETITION}.zip') as archive:
            for name in names:
                matches = [s for s in archive.namelist() if Path(s).name == name]
                if len(matches) != 1:
                    raise ValueError(f'Expected exactly one archive member for {name}')
                # Write only the named data files, never an existing submission CSV.
                (data_dir / name).write_bytes(archive.read(matches[0]))
    train = pd.read_csv(data_dir / names[0])
    test = pd.read_csv(data_dir / names[1])
    assert set(train.columns) == set(FEATURES + ['ID', 'Y'])
    assert set(test.columns) == set(FEATURES + ['ID'])
    assert train.ID.is_unique and test.ID.is_unique
    assert set(train.ID).isdisjoint(test.ID)
    assert np.isfinite(train[FEATURES + ['Y']].to_numpy()).all()
    assert np.isfinite(test[FEATURES].to_numpy()).all()
    assert (train[FEATURES].std() > 0).all()
    return train, test


def check_kaggle_access():
    """Use the Week 1 authentication route without uploading predictions."""
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    response = api.competition_list_files(COMPETITION)
    status = {'competition': COMPETITION, 'competition_files_accessible': True,
              'files': [item.name for item in response.files],
              'upload_attempted': False}
    print(json.dumps(status, indent=2))
    return status


def submit_predictions(data_dir, output, message):
    """Upload saved predictions; the API response does not verify a score."""
    from datetime import datetime, timezone

    submission_path = output/'submission.csv'
    manifest = json.loads((output/'run_manifest.json').read_text())
    if sha256(submission_path) != manifest['submission_sha256']:
        raise ValueError('Submission differs from the checked analysis output.')
    submission = pd.read_csv(submission_path)
    _, test = get_data(data_dir)
    if sha256(data_dir/'test_data.csv') != manifest['input_sha256']['test_data.csv']:
        raise ValueError('Test data differ from the data used for these predictions.')
    if (list(submission.columns) != ['ID', 'Y']
            or not submission.ID.equals(test.ID)
            or not np.isfinite(submission.Y.to_numpy()).all()):
        raise ValueError('Submission columns, IDs or predictions are invalid.')
    from kaggle.api.kaggle_api_extended import KaggleApi

    api = KaggleApi()
    api.authenticate()
    response = api.competition_submit(file_name=str(submission_path.resolve()),
                                      message=message, competition=COMPETITION)
    receipt = {'competition': COMPETITION,
               'requested_at_utc': datetime.now(timezone.utc).isoformat(),
               'selected_model': manifest['selected_model'],
               'submission_sha256': sha256(submission_path),
               'api_response': str(response), 'score_verified': False,
               'public_score': None}
    json_write(output/'kaggle_submission_receipt.json', receipt)
    print('Kaggle upload request returned. Check the submission status and score.')
    print(response)
    return receipt


def fit_quadratic_lasso(x, y, label, audit):
    """All degree-2 terms, training-only scaling and Week 4 AICc selection."""
    started = time.perf_counter()
    polynomial = PolynomialFeatures(degree=2, include_bias=False)
    design = polynomial.fit_transform(x)
    scaler = StandardScaler().fit(design)
    z = np.asfortranarray(scaler.transform(design))
    walker = Lasso(warm_start=True, max_iter=20000, tol=1e-4)
    path = []
    best_value, best_alpha = np.inf, None
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning)
        for alpha in ALPHAS:
            walker.set_params(alpha=float(alpha))
            walker.fit(z, y)
            k = int(np.count_nonzero(walker.coef_)) + 1
            error = mse(y, walker.predict(z))
            # k approximates model degrees of freedom, as in the lecture.
            value = np.inf
            if k < len(y) - 1 and error > 0:
                value = len(y) * np.log(error) + 2*k + 2*k*(k+1)/(len(y)-k-1)
            path.append({'alpha': float(alpha), 'nonzero': k-1,
                         'AICc': float(value) if np.isfinite(value) else None})
            if value < best_value:
                best_value, best_alpha = float(value), float(alpha)
        if best_alpha is None:
            raise RuntimeError(f'No admissible AICc fit: {label}')
        model = Lasso(alpha=best_alpha, max_iter=20000, tol=1e-4).fit(z, y)
    record = {'fit': label, 'n': len(y), 'raw_predictors': x.shape[1],
              'candidate_terms': z.shape[1], 'alpha': best_alpha,
              'nonzero': int(np.count_nonzero(model.coef_)), 'AICc': best_value,
              'grid_boundary': bool(best_alpha in (ALPHAS[0], ALPHAS[-1])),
              'seconds': time.perf_counter()-started, 'path': path}
    audit.append(record)
    print(f"  {label}: alpha={best_alpha:.6g}; terms={record['nonzero']}; "
          f"{record['seconds']:.1f}s", flush=True)
    return {'polynomial': polynomial, 'scaler': scaler, 'model': model}


def predict_quadratic(fit, x):
    return fit['model'].predict(fit['scaler'].transform(fit['polynomial'].transform(x)))


def fit_controls(x, y, seed, label, audit):
    """Week 8 cross-fitted residual regression, followed by nuisance-model refits.

    theta = sum((D-m_oof)*(Y-l_oof)) / sum((D-m_oof)**2).
    The anonymous data do not establish a causal interpretation.
    """
    d, w = x[:, D_INDEX], x[:, W_INDICES]
    l_oof, m_oof = np.full(len(y), np.nan), np.full(len(y), np.nan)
    covered = np.zeros(len(y), dtype=int)
    cf_id = np.full(len(y), -1)
    for fold, (tr, va) in enumerate(
            KFold(CROSSFIT_FOLDS, shuffle=True, random_state=seed).split(w), 1):
        assert len(np.intersect1d(tr, va)) == 0
        outcome = fit_quadratic_lasso(w[tr], y[tr], f'{label}/CF{fold}/Y', audit)
        treatment = fit_quadratic_lasso(w[tr], d[tr], f'{label}/CF{fold}/D', audit)
        l_oof[va] = predict_quadratic(outcome, w[va])
        m_oof[va] = predict_quadratic(treatment, w[va])
        covered[va] += 1
        cf_id[va] = fold
    assert (covered == 1).all()
    assert np.isfinite(l_oof).all() and np.isfinite(m_oof).all()
    u, v = y-l_oof, d-m_oof
    denominator = float(v @ v)
    if denominator <= np.finfo(float).eps:
        raise ValueError('X34 has no residual variation after controls.')
    theta = float(v @ u / denominator)
    check = LinearRegression(fit_intercept=False).fit(v[:, None], u)
    assert np.isclose(theta, check.coef_[0], rtol=1e-10, atol=1e-10)
    outcome = fit_quadratic_lasso(w, y, f'{label}/full/Y', audit)
    treatment = fit_quadratic_lasso(w, d, f'{label}/full/D', audit)
    return {'outcome': outcome, 'treatment': treatment, 'theta': theta,
            'residual_variance': float(np.mean(v**2)),
            'residuals': pd.DataFrame({'Y': y, 'D': d, 'l_oof': l_oof,
                                     'm_oof': m_oof, 'residual_Y': u,
                                     'residual_D': v, 'crossfit_fold': cf_id})}


def predict_controls(fit, x):
    w, d = x[:, W_INDICES], x[:, D_INDEX]
    # Add the outcome component back: residual predictions alone are not Y.
    return (predict_quadratic(fit['outcome'], w)
            + fit['theta'] * (d-predict_quadratic(fit['treatment'], w)))


def run_analysis(data_dir, output):
    started = time.perf_counter()
    train, test = get_data(data_dir)
    x, y = train[FEATURES].to_numpy(), train.Y.to_numpy()
    x_test = test[FEATURES].to_numpy()
    d, w = x[:, D_INDEX], x[:, W_INDICES]
    naive = LinearRegression().fit(d[:, None], y)
    controlled = LinearRegression().fit(x, y)
    residual_y = y-LinearRegression().fit(w, y).predict(w)
    residual_d = d-LinearRegression().fit(w, d).predict(w)
    fwl_theta = float(residual_d @ residual_y / (residual_d @ residual_d))
    assert np.isclose(fwl_theta, controlled.coef_[D_INDEX], atol=1e-10)
    association = pd.DataFrame({
        'model': ['X34 alone', 'X34 + 49 linear controls', 'FWL residual regression'],
        'X34_slope': [naive.coef_[0], controlled.coef_[D_INDEX], fwl_theta]})
    association.to_csv(output / 'association_coefficients.csv', index=False)

    predictions = np.full((len(y), len(MODEL_NAMES)), np.nan)
    fold_id, covered = np.full(len(y), -1), np.zeros(len(y), dtype=int)
    fit_audit, fold_details = [], []
    for fold, (tr, va) in enumerate(
            KFold(OUTER_FOLDS, shuffle=True, random_state=SEED).split(x), 1):
        print(f'OUTER FOLD {fold}/{OUTER_FOLDS}: train={len(tr)}, validation={len(va)}', flush=True)
        fold_id[va], covered[va] = fold, covered[va]+1
        predictions[va, 0] = LinearRegression().fit(x[tr, D_INDEX, None], y[tr]).predict(x[va, D_INDEX, None])
        predictions[va, 1] = LinearRegression().fit(x[tr], y[tr]).predict(x[va])
        baseline = fit_quadratic_lasso(x[tr], y[tr], f'outer{fold}/baseline', fit_audit)
        predictions[va, 2] = predict_quadratic(baseline, x[va])
        cf = fit_controls(x[tr], y[tr], SEED+100+fold, f'outer{fold}', fit_audit)
        predictions[va, 3] = predict_controls(cf, x[va])
        fold_details.append({'fold': fold, 'train_n': len(tr), 'validation_n': len(va),
                             'control_slope': cf['theta'],
                             'residual_D_variance': cf['residual_variance']})
        print('  VALIDATION:', {name: round(mse(y[va], predictions[va, j]), 6)
                                for j, name in enumerate(MODEL_NAMES)}, flush=True)
    assert (covered == 1).all() and np.isfinite(predictions).all()
    oof = pd.DataFrame(predictions, columns=MODEL_NAMES)
    oof.insert(0, 'fold', fold_id); oof.insert(0, 'Y', y); oof.insert(0, 'ID', train.ID)
    oof.to_csv(output / 'oof_predictions.csv', index=False)
    fold_scores = pd.DataFrame([
        {'model': name, 'fold': fold,
         'MSE': mse(y[fold_id == fold], predictions[fold_id == fold, j])}
        for j, name in enumerate(MODEL_NAMES) for fold in range(1, OUTER_FOLDS+1)])
    scores = pd.DataFrame([
        {'model': name, 'OOF_MSE': mse(y, predictions[:, j]),
         'OOF_R2': 1-mse(y, predictions[:, j])/np.var(y),
         'fold_MSE_sd': float(fold_scores.loc[fold_scores.model == name, 'MSE'].std(ddof=1))}
        for j, name in enumerate(MODEL_NAMES)]).sort_values('OOF_MSE', kind='stable').reset_index(drop=True)
    scores.to_csv(output / 'model_comparison.csv', index=False)
    fold_scores.to_csv(output / 'fold_mse.csv', index=False)
    pd.DataFrame(fold_details).to_csv(output / 'fold_details.csv', index=False)
    paired = fold_scores.pivot(index='fold', columns='model', values='MSE')
    paired['baseline_minus_controls'] = paired[MODEL_NAMES[2]]-paired[MODEL_NAMES[3]]
    paired.to_csv(output / 'paired_fold_comparison.csv')
    winner = str(scores.iloc[0]['model'])
    print('\nRESULTS\n'+scores.to_string(index=False), flush=True)
    print('SELECTED:', winner, flush=True)

    # Full-sample control fit supplies the descriptive slope, even if not selected.
    # Full-sample baseline is also retained as an explicit single-model comparator.
    full_baseline = fit_quadratic_lasso(x, y, 'final/baseline', fit_audit)
    full_controls = fit_controls(x, y, SEED+999, 'final', fit_audit)
    full_controls['residuals'].assign(ID=train.ID).to_csv(output / 'full_sample_crossfit_residuals.csv', index=False)
    final_predictions = {
        MODEL_NAMES[0]: naive.predict(x_test[:, D_INDEX, None]),
        MODEL_NAMES[1]: controlled.predict(x_test),
        MODEL_NAMES[2]: predict_quadratic(full_baseline, x_test),
        MODEL_NAMES[3]: predict_controls(full_controls, x_test),
    }
    for name, filename in [(winner, 'submission.csv'),
                           (MODEL_NAMES[2], 'submission_quadratic_lasso.csv'),
                           (MODEL_NAMES[3], 'submission_controls.csv')]:
        prediction = final_predictions[name]
        assert len(prediction) == len(test) and np.isfinite(prediction).all()
        result = pd.DataFrame({'ID': test.ID, 'Y': prediction})
        result.to_csv(output / filename, index=False)
        saved = pd.read_csv(output / filename)
        assert saved.ID.equals(test.ID) and list(saved.columns) == ['ID', 'Y']
        assert np.allclose(saved.Y, prediction, atol=1e-12, rtol=1e-12)
    fit = full_baseline
    coefficients = pd.DataFrame({
        'term': fit['polynomial'].get_feature_names_out(FEATURES),
        'standardised_coefficient': fit['model'].coef_,
        'raw_scale_coefficient': fit['model'].coef_/fit['scaler'].scale_})
    coefficients.to_csv(output / 'quadratic_lasso_coefficients.csv', index=False)
    json_write(output / 'fit_audit.json', fit_audit)
    manifest = {
        'competition': COMPETITION, 'data_directory': str(data_dir.resolve()),
        'training_n': len(train), 'test_n': len(test), 'focal_variable': FOCAL,
        'control_variables': [FEATURES[i] for i in W_INDICES],
        'outer_folds': OUTER_FOLDS, 'crossfit_folds': CROSSFIT_FOLDS, 'seed': SEED,
        'alphas': ALPHAS.tolist(), 'penalty_selection': 'AICc',
        'selected_model': winner, 'selected_oof_mse': float(scores.iloc[0].OOF_MSE),
        'selected_oof_r2': float(scores.iloc[0].OOF_R2),
        'naive_slope': float(naive.coef_[0]), 'controlled_ols_slope': fwl_theta,
        'crossfit_slope_full_sample': full_controls['theta'],
        'control_gain_over_quadratic': float(paired.baseline_minus_controls.mean()),
        'controls_win_folds': int((paired.baseline_minus_controls > 0).sum()),
        'quadratic_terms': int(len(coefficients)),
        'quadratic_nonzero': int(np.count_nonzero(fit['model'].coef_)),
        'fit_count': len(fit_audit),
        'alpha_boundary_count': sum(a['grid_boundary'] for a in fit_audit),
        'convergence_warnings': 0, 'kaggle_submitted': False,
        'input_sha256': {name: sha256(data_dir/name) for name in ['train_data.csv', 'test_data.csv']},
        'submission_sha256': sha256(output/'submission.csv'),
        'analysis_code_sha256': sha256(Path(__file__)),
        'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                     'pandas': pd.__version__, 'sklearn': sklearn.__version__},
        'elapsed_seconds': time.perf_counter()-started,
        'validation_note': 'Candidate-level OOF estimates; choosing the winner may introduce optimism. '
                           'X34 was originally selected using this dataset in A1. A3 scores are historical.'
    }
    json_write(output / 'run_manifest.json', manifest)
    print(f"Completed in {manifest['elapsed_seconds']:.1f}s. No Kaggle upload performed.", flush=True)
    return manifest


def build_report(output):
    """Rebuild figures and the review PDF from saved, checked numerical results."""
    import html
    import os
    plot_cache = output.resolve()/'.matplotlib_cache'
    plot_cache.mkdir(exist_ok=True)
    os.environ.setdefault('MPLCONFIGDIR', str(plot_cache))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from reportlab.lib import colors
    from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak)

    m = json.loads((output/'run_manifest.json').read_text())
    scores = pd.read_csv(output/'model_comparison.csv')
    oof = pd.read_csv(output/'oof_predictions.csv')
    paired = pd.read_csv(output/'paired_fold_comparison.csv')
    associations = pd.read_csv(output/'association_coefficients.csv')
    for row in scores.itertuples(index=False):
        assert np.isclose(mse(oof.Y, oof[row.model]), row.OOF_MSE, atol=1e-10)
    assert sha256(output/'submission.csv') == m['submission_sha256']
    lookup = scores.set_index('model').OOF_MSE
    baseline_mse, control_mse = float(lookup[MODEL_NAMES[2]]), float(lookup[MODEL_NAMES[3]])
    gain = baseline_mse-control_mse
    winner = m['selected_model']
    winner_prose = {'Quadratic LASSO': 'Quadratic LASSO',
                    'Cross-fitted constant slope': 'The constant-slope control model',
                    'OLS: all 50 variables': 'OLS with all 50 variables',
                    'OLS: X34 alone': 'OLS with X34 alone'}[winner]
    paragraphs = [
        "Building on Assignment 3's quadratic LASSO, we ask whether Week 8 residual-based control adjustment improves prediction. X34 is retained from Assignment 1. We compare four individual models without stacking.",
        f"Controlling linearly for the remaining 49 variables changes the X34 OLS slope from {m['naive_slope']:.3f} to {m['controlled_ols_slope']:.3f}. FWL residual regression reproduces the controlled coefficient. This demonstrates sensitivity to the included controls, not removal of all confounding or identification of a causal effect.",
        "The baseline uses all 50 variables, their squares and pairwise interactions, with standardisation and AICc-selected LASSO penalties. The control specification predicts Y and X34 from the other variables using quadratic LASSO. Four-fold cross-fitting produces held-out residuals, which are regressed using one shared slope. Predictions restore the outcome component explained by controls. This implements the partialling-out form of double machine learning.",
        f"On identical five-fold outer validation splits, baseline MSE is {baseline_mse:.3f} versus {control_mse:.3f} for the control specification, which improves {m['controls_win_folds']} of five folds. Both models already use the other 49 variables. The control specification excludes X34's square and interactions with controls, so this comparison changes functional form and estimation, not simply whether controls are included.",
        f"Scaling and penalty selection exclude scored observations. Selecting the lowest validation error can be optimistic, especially after earlier assignments used the same data. These results do not establish a trade-off between causal identification and prediction. A3's historical stack scored 6.023 and was not rerun. {winner_prose} is refitted on all {m['training_n']:,} training observations for {m['test_n']:,} test predictions.",
        'The corresponding Kaggle score and screenshot remain pending; no new public score is claimed.'
    ]
    word_count = len(' '.join(paragraphs).split())
    assert word_count <= 270, word_count

    figures = output/'figures'; figures.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 11,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, ax = plt.subplots(figsize=(8.0, 4.35))
    ordered = scores.iloc[::-1]
    short_labels = {'OLS: X34 alone': 'OLS: X34 alone',
                    'OLS: all 50 variables': 'OLS: all 50',
                    'Quadratic LASSO': 'Quadratic LASSO',
                    'Cross-fitted constant slope': 'Cross-fitted controls'}
    bars = ax.barh([short_labels[s] for s in ordered.model], ordered.OOF_MSE,
                   color=['#2077a9' if s == winner else '#849cac' for s in ordered.model], height=.55)
    ax.bar_label(bars, labels=[f'{v:.3f}' for v in ordered.OOF_MSE], padding=5, fontsize=11)
    ax.set_xlim(0, float(scores.OOF_MSE.max())*1.13)
    ax.set_xlabel('Out-of-fold mean squared error (lower is better)')
    ax.set_title('Single-model prediction on the same five folds', pad=15)
    ax.grid(axis='x', color='#dddddd', alpha=.55); ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(figures/'model_comparison.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    improvements = paired.baseline_minus_controls.to_numpy()
    bars = ax.bar(paired['fold'], improvements,
                  color=['#2077a9' if value > 0 else '#c06a4b' for value in improvements], width=.55)
    ax.axhline(0, color='black', linewidth=.8)
    ax.set_xticks(range(1, OUTER_FOLDS+1))
    ax.set_xlabel('Outer validation fold')
    ax.set_ylabel('Baseline MSE minus control-model MSE')
    ax.set_title('Does control adjustment improve prediction?', pad=15)
    lo, hi = min(0, float(improvements.min())), max(0, float(improvements.max()))
    pad = max((hi-lo)*.25, .04)
    ax.set_ylim(lo-pad, hi+pad)
    ax.bar_label(bars, labels=[f'{v:+.3f}' for v in improvements], padding=4)
    fig.tight_layout()
    fig.savefig(figures/'paired_fold_gains.png', dpi=240, bbox_inches='tight')
    plt.close(fig)

    # Use the same Times New Roman typography as the supplied A3 report.
    font_dir = Path('/System/Library/Fonts/Supplemental')
    if (font_dir/'Times New Roman.ttf').is_file():
        for name, filename in [('ReportTimes', 'Times New Roman.ttf'),
                               ('ReportTimes-Bold', 'Times New Roman Bold.ttf'),
                               ('ReportTimes-Italic', 'Times New Roman Italic.ttf')]:
            pdfmetrics.registerFont(TTFont(name, str(font_dir/filename)))
        pdfmetrics.registerFontFamily('ReportTimes', normal='ReportTimes',
                                      bold='ReportTimes-Bold', italic='ReportTimes-Italic')
        regular, bold, italic = 'ReportTimes', 'ReportTimes-Bold', 'ReportTimes-Italic'
    else:
        regular, bold, italic = 'Times-Roman', 'Times-Bold', 'Times-Italic'
    body = ParagraphStyle('Body', fontName=regular, fontSize=11, leading=14,
                          alignment=TA_JUSTIFY, spaceAfter=7)
    heading = ParagraphStyle('Heading', parent=body, fontName=bold, fontSize=15.4,
                             leading=19, spaceBefore=6, spaceAfter=5, alignment=TA_LEFT)
    caption = ParagraphStyle('Caption', parent=body, alignment=TA_CENTER, leading=13.4)
    small = ParagraphStyle('Small', parent=body, fontSize=9.2, leading=12)
    prompt_style = ParagraphStyle('Prompt', parent=small, fontSize=9.2,
                                  leading=12, alignment=TA_LEFT)
    title = ParagraphStyle('Title', parent=body, fontName=bold, fontSize=16,
                           leading=20, alignment=TA_CENTER)
    subtitle = ParagraphStyle('Subtitle', parent=title, fontSize=14, leading=18)
    draft_style = ParagraphStyle('Draft', parent=small, fontName=italic, alignment=TA_CENTER)
    pdf_dir = output/'pdf'; pdf_dir.mkdir(exist_ok=True)
    pdf_path = pdf_dir/'Assignment4_Report_revised.pdf'
    width = A4[0]-125
    doc = SimpleDocTemplate(str(pdf_path), pagesize=A4,
                           leftMargin=62.5, rightMargin=62.5, topMargin=60, bottomMargin=54,
                           title='Assignment 4: Controls and Single-Model Prediction',
                           author='Bindi Wang (Jason), Mingqian Zhang, Tianxing Ma, Qingyu Xie')
    story = []

    def p(text, style=body):
        return Paragraph(html.escape(str(text)), style)

    def table(headers, rows, widths, highlight=None):
        data = [[Paragraph(html.escape(str(v)), ParagraphStyle(
                    'TableHeader', parent=body, fontName=bold, alignment=TA_LEFT, spaceAfter=0))
                 for v in headers]]
        data += [[p(v, ParagraphStyle('Cell', parent=body, alignment=TA_LEFT if j == 0 else 2,
                                      fontName=bold if i == highlight else regular, spaceAfter=0))
                  for j, v in enumerate(row)] for i, row in enumerate(rows, 1)]
        t = Table(data, colWidths=widths, hAlign='CENTER')
        commands = [('LINEABOVE', (0, 0), (-1, 0), .55, colors.black),
                    ('LINEBELOW', (0, 0), (-1, 0), .4, colors.black),
                    ('LINEBELOW', (0, -1), (-1, -1), .55, colors.black),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('TOPPADDING', (0, 0), (-1, -1), 4),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 4)]
        if highlight is not None:
            commands.append(('BACKGROUND', (0, highlight), (-1, highlight), colors.HexColor('#edf3f6')))
        t.setStyle(TableStyle(commands))
        return t

    def figure(path, max_height):
        image = Image(str(path))
        scale = min(width/image.imageWidth, max_height/image.imageHeight)
        image.drawWidth, image.drawHeight = image.imageWidth*scale, image.imageHeight*scale
        image.hAlign = 'CENTER'
        return image

    story += [p('ECOM90025 Advanced Data Analysis', title), Spacer(1, 12),
              p('Assignment 4: Controls and Single-Model Prediction', subtitle),
              p('Review draft - Kaggle evidence and public code link pending', draft_style), Spacer(1, 15)]
    members = [['Name', 'Student ID'], ['Bindi Wang (Jason)', '1342167'],
               ['Mingqian Zhang', '1785849'], ['Tianxing Ma', '1826351'], ['Qingyu Xie', '1777098']]
    member_table = Table(members, colWidths=[125, 75], hAlign='CENTER')
    member_table.setStyle(TableStyle([('FONTNAME', (0, 0), (-1, -1), regular),
                                      ('FONTNAME', (0, 0), (-1, 0), bold),
                                      ('FONTSIZE', (0, 0), (-1, -1), 11),
                                      ('LEADING', (0, 0), (-1, -1), 12.5),
                                      ('TOPPADDING', (0, 0), (-1, -1), 1),
                                      ('BOTTOMPADDING', (0, 0), (-1, -1), 1)]))
    story += [member_table, Spacer(1, 13), p('1. Control variables', heading),
              p(paragraphs[0]), p(paragraphs[1]), p('2. Prediction without stacking', heading),
              p(paragraphs[2]), p(paragraphs[3]), p(paragraphs[4]),
              p('3. Kaggle result', heading), p(paragraphs[5]), PageBreak()]

    rows = [[row.model, f'{row.OOF_MSE:.6f}', f'{row.OOF_R2:.6f}']
            for row in scores.itertuples(index=False)]
    story += [table(['Model', 'CV MSE', 'CV R²'], rows, [width-153, 82, 71], 1), Spacer(1, 9),
              p('Table 1: Pooled out-of-fold predictions for 2,400 observations on five identical shuffled folds (seed 1; 480 validation observations per fold). Lower MSE is better. R² = 1 - pooled MSE / Var(Y). The highlighted row is selected for the revised prediction file.', caption),
              Spacer(1, 25), figure(figures/'model_comparison.png', 295), Spacer(1, 8),
              p('Figure 1: Validation error of the four individual candidate models. All preprocessing and AICc penalty selection use only the corresponding training data. The A3 stack is historical context and is not one of these candidates.', caption),
              Spacer(1, 18),
              p(f"The quadratic baseline has {m['quadratic_terms']:,} candidate terms: 50 levels, 50 squares and 1,225 distinct pairwise interactions. Its full-sample fit retains {m['quadratic_nonzero']} nonzero coefficients, excluding the intercept.", small),
              PageBreak()]

    slope_rows = [[row.model, f'{row.X34_slope:.6f}'] for row in associations.itertuples(index=False)]
    slope_rows.append(['Cross-fitted constant slope', f"{m['crossfit_slope_full_sample']:.6f}"])
    story += [table(['Specification', 'X34 slope'], slope_rows, [width-115, 115]), Spacer(1, 9),
              p('Table 2: Descriptive X34 coefficients using all 2,400 training observations. The first three rows use linear OLS controls; the final row uses quadratic LASSO controls and four-fold cross-fitting. FWL exactly reproduces controlled OLS. The cross-fitted coefficient comes from a different specification and need not equal it. None is interpreted causally.', caption),
              Spacer(1, 24), figure(figures/'paired_fold_gains.png', 290), Spacer(1, 8),
              p(f"Figure 2: Paired differences in validation MSE between quadratic LASSO and cross-fitted controls. Positive values favour controls. Controls improve {m['controls_win_folds']} of five folds; the pooled gain is {gain:+.6f}. Training sets overlap, so this figure is descriptive and does not supply an independent-sample confidence interval.", caption),
              Spacer(1, 18),
              p('Scope note: Cross-fitting residualises only Y and X34. The other 49 variables enter the two nuisance regressions through fixed quadratic dictionaries. No squared or cubic X34 terms are residualised and no varying-slope interactions are fitted.', small), PageBreak()]

    code_text = 'Complete Python code: public link pending.'
    declaration = [
        'OpenAI Codex assisted with checking the assignment requirements and supplied lectures, proposing and implementing the simplified model comparison, running validation, checking numerical outputs, preparing figures and predictions, and drafting this report. The authors requested simpler methods and comparison without stacking. Codex proposed the remaining specifications; these are not represented as independently devised by the authors.',
        'The supplied A3 report disclosed earlier Claude assistance and served as historical context and a layout reference. A Gemini-generated comparison was consulted briefly during report review. The authors must review the final methods and interpretations.',
        'Material prompts are reproduced below. Prompts originally submitted in Chinese are labelled as English translations; original English instructions are identified separately. The supplied materials included the A3/A4 requirements, earlier reports and Python scripts, and the lecture notebooks. The English-only revision changes the language and submission utilities, not the fitted models or prediction results.'
    ]
    prompts = [
        '1. English translation: "Complete Assignment 4. Check whether the current report meets the requirements and uses methods beyond the lectures. We used stacking in A3 but could not explain why the weights were unequal, could be negative, or did not sum to one, or explain its formula and mechanism. For unfamiliar methods, the lecturer expects us to understand what they do and why. A4 may omit stacking or compare it. We have learned controls and can include them in the analysis and report."',
        'Original English workflow instructions: "Ask me questions if needed. Give some python code first, and then wait for my instruction before writing. /grill-me. Use kaggle API key the similar way as in LEC/ W1/Week01_intro.ipynb"; "write a report first, once I let you to submit the result, I’ll give you the screenshot and github link"; "make sure all the methods we used are in the lectures"; "submit the result to kaggle, and write the AI statement like we did in A3".',
        '2. English translation: "Selecting 12 variables by coefficient mass before generating cubic terms was used in A3, but this exact selection rule was not found in the checked lectures. Residualising D, D squared and D cubed, combined with 49 slope interactions, also extends the classroom examples and requires further derivation. Can we omit these two constructions? Is there a better approach? What if we do not use stacking?"',
        '3. English translation: "Run the complete analysis, produce a report in the same format as the A3 report, and provide the code. I want to see the results."',
        '4. English translation: "Find lecture support for everything used in this report and its code, and explain the underlying logic and concepts in connection with the lectures."',
        '5. English translation: "No Chinese is allowed in the final reports or code. Check that the notebook and report correspond, explain why the code is shorter than A3, and identify any Kaggle API or code-publication issues. Summarise everything still missing so I can prepare it. Follow the A3 report and requirements, and translate the report prompts into English."'
    ]
    story += [p('4. Code', heading), p(code_text),
              p('5. AI Use Declaration', heading)]
    story += [p(text, small) for text in declaration]
    story += [p(text, prompt_style) for text in prompts]
    source_text = ('Course sources: Week03_regression.ipynb (OLS, interactions and K-fold validation); '
                   'Week04_LASSO.ipynb (standardisation, LASSO and AICc); '
                   'Week08_controls.ipynb (FWL, constant-slope residualisation and cross-fitting); '
                   'Assignment3_Report.pdf (historical benchmark and layout); Assignment01.ipynb (submission requirements).')
    story += [Spacer(1, 5), p(source_text, small),
              p(f'Report body: {word_count} words. Body plus Code prose: {word_count+len(code_text.split())} words; headings and the separate declaration excluded.', small)]

    def footer(canvas, document):
        canvas.saveState(); canvas.setFont(regular, 10)
        canvas.drawCentredString(A4[0]/2, 25, str(document.page)); canvas.restoreState()

    doc.build(story, onFirstPage=footer, onLaterPages=footer)
    markdown = ('# Assignment 4: Controls and Single-Model Prediction\n\n'
                'Review draft: Kaggle evidence and public code link pending.\n\n'
                '## 1. Control variables\n\n'+'\n\n'.join(paragraphs[:2])
                +'\n\n## 2. Prediction without stacking\n\n'+'\n\n'.join(paragraphs[2:5])
                +'\n\n## 3. Kaggle result\n\n'+paragraphs[5]
                +'\n\n## 4. Code\n\n'+code_text
                +'\n\n## 5. AI Use Declaration\n\n'+'\n\n'.join(declaration+prompts)
                +'\n\n'+source_text+f'\n\nReport body: {word_count} words.\n')
    (output/'Assignment4_Report_revised.md').write_text(markdown, encoding='utf-8')
    json_write(output/'report_manifest.json', {
        'body_words': word_count, 'body_plus_code_words': word_count+len(code_text.split()),
        'pdf': str(pdf_path.resolve()), 'pdf_sha256': sha256(pdf_path),
        'code_sha256': sha256(Path(__file__)),
        'kaggle_score': None, 'public_code_url': None, 'status': 'results draft'})
    print(f'Report: {pdf_path}\nBody words: {word_count}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = HERE.parent/'data' if (HERE.parent/'data'/'train_data.csv').is_file() else HERE/'data'
    parser.add_argument('--data-dir', type=Path, default=default_data)
    parser.add_argument('--output-dir', type=Path, default=HERE/'output')
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument('--analysis-only', action='store_true')
    modes.add_argument('--report-only', action='store_true')
    modes.add_argument('--check-kaggle', action='store_true',
                       help='Check competition access without uploading or training.')
    modes.add_argument('--submit-to-kaggle', action='store_true',
                       help='Upload the saved checked submission.csv without retraining.')
    parser.add_argument('--submission-message', default='Assignment 4 revised: quadratic LASSO',
                        help='Description attached to an explicitly requested Kaggle upload.')
    args = parser.parse_args(argv)
    if args.check_kaggle:
        check_kaggle_access()
        return
    if args.submit_to_kaggle:
        submit_predictions(args.data_dir, args.output_dir, args.submission_message)
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=1):
        if not args.report_only:
            run_analysis(args.data_dir, args.output_dir)
    if not args.analysis_only:
        build_report(args.output_dir)


if __name__ == '__main__':
    main()

