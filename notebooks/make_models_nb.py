"""
notebooks/make_models_nb.py  (Persons 2 and 3)

Builds AND executes notebooks/02_models.ipynb: a walk-through of the model comparison, the Bayesian optimisation
and the HMM results. The notebook only READS the outputs of the experiment scripts (README section 6); it does
not retrain anything. Each code cell is really executed here and its output (text, tables, figures) is stored in
the .ipynb, so the notebook shows actual results even without Jupyter installed.

    python notebooks/make_models_nb.py      (from the repo root)
"""
import base64
import contextlib
import io
import json
import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT); sys.path.insert(0, str(ROOT))

CELLS = []
md = lambda s: CELLS.append(("markdown", s.strip("\n")))
code = lambda s: CELLS.append(("code", s.strip("\n")))

# --------------------------------------------------------------------------------------------- content
md("""
# 02 — Models, Bayesian optimisation and HMM (Persons 2 and 3)

This notebook **reads** the outputs of the experiment scripts; it does not retrain. To reproduce from scratch, run in order
(repo root): `python -m src.features.dataset`, `python -m src.models.train_compare`, `python -m src.tuning.bayes_opt 30`,
`python -m src.hmm.evaluate 30`, `python -m src.models.timing`, `python -m src.models.route_eval`,
`python -m src.models.error_analysis`, `python -m src.hmm.break_analysis`.

**Partitions:** train (78 clean trips) for every fit and choice, using GroupKFold(5) by trip; trips 33 and 54 (label-suspect) are reported separately;
student_test (10 trips) is used only to report. The evaluator partition is never read.
""")
code("""
import os, sys
if os.path.basename(os.getcwd()) == 'notebooks':
    os.chdir('..')
sys.path.insert(0, '.')
try:
    show                                   # provided by notebooks/make_models_nb.py when it executes this notebook
except NameError:                          # in Jupyter
    from IPython.display import Image, display
    def show(path): display(Image(path))
import pandas as pd, numpy as np, json, yaml
pd.set_option('display.width', 220); pd.set_option('display.max_columns', 40); pd.set_option('display.precision', 4)
from src.features.build import FEATURE_COLUMNS
print(len(FEATURE_COLUMNS), 'features:', ', '.join(FEATURE_COLUMNS))
""")
md("""
## 1. Dataset for the classifier
One row per (GPS point, candidate arc); label = 1 if the candidate is the reference arc (P1's monotone-DP alignment).
""")
code("""
pts = {p: pd.read_parquet(f'data/processed/{p}_points.parquet') for p in ['train', 'student_test']}
F = pd.read_parquet('data/processed/student_test_features.parquet')
for p, d in pts.items():
    c = d[~d.trip_label_suspect]
    print(f"{p:13s} trips={d.trip_id.nunique():3d} points={len(d):7,d} candidate miss rate: all={1-d.label_in_candidates.mean():.4f} clean={1-c.label_in_candidates.mean():.4f}")
print('student_test candidate rows:', f'{len(F):,}', ' positive share:', round(F.label.mean(), 4))
F.groupby('label')[['dist_m', 'cos_c', 'next_conn_dist', 'win_frac', 'speed']].mean()
""")
md("""
The true arcs (label 1) are much closer, aligned with the direction of motion (`cos_c` ≈ +0.8: the **sign** separates the two
directions of a street), and connected to close candidates at the next point.

## 2. Model comparison (Section 3.1)
""")
code("""
T = pd.read_csv('results/model_comparison_table.csv')
T
""")
code("""
C = pd.read_csv('results/model_comparison.csv')
C[['model', 'cv_top1_mean', 'cv_top1_std', 'cv_macro_f1_mean', 'cv_log_loss_mean', 'cv_train_top1_mean',
   'test_top1', 'test_macro_f1', 'test_log_loss', 'suspect_trips_top1']].sort_values('cv_top1_mean', ascending=False)
""")
md("### Per-fold results (fold spread vs model gap)")
code("""
folds = pd.read_csv('results/model_cv_folds.csv')
folds.pivot(index='fold', columns='model', values='top1').round(4)
""")
md("""
Every model finds the same folds hard (fold 2 is easiest for all of them). The fold-to-fold spread (≈ 0.035–0.038) is larger than the
gap between the four best models, so CV alone does not separate RF, LightGBM and MLP (see report 3.4 for the selection argument).
""")
code("show('results/figures/models_comparison.png')")
code("show('results/figures/models_train_cv_test.png')")
md("""
Train-fit vs CV vs test: Random Forest has the largest gap (0.948 vs 0.869: high variance), logistic regression and naive Bayes
almost none, but they score low (bias).
""")
code("show('results/figures/models_confusion_matrices.png')")
md("Only the MLP has an epoch history, so it is the only model with a loss curve:")
code("""
lc = pd.read_csv('results/loss_curve_mlp.csv')
print('best validation accuracy at epoch', int(lc.loc[lc.val_score.idxmax(), 'epoch']), '; training stopped at epoch', int(lc.epoch.max()))
show('results/figures/models_mlp_loss_curve.png')
""")
md("## 3. Inference time (idle-machine benchmark)")
code("pd.read_csv('results/inference_timing.csv')[['model', 'seconds', 'ms_per_1k_rows', 'model_file_mb']]")
md("""
## 4. Bayesian optimisation of LightGBM (Section 4)
Optuna TPE, 30 trials, objective = 3-fold group-CV log loss; trial 0 = the untuned defaults.
""")
code("""
tr = pd.read_csv('results/tuning/optuna_trials.csv')
print(tr.state.value_counts().to_dict())
print(json.load(open('results/tuning/best_params.json'))['best_params'])
tr[['number', 'value', 'state', 'user_attrs_top1', 'params_n_estimators', 'params_learning_rate', 'params_num_leaves',
    'params_max_depth', 'params_min_child_samples']].sort_values('value').head(8)
""")
code("show('results/figures/tuning_history.png')")
code("show('results/figures/tuning_importance.png')")
code("pd.read_csv('results/tuning/untuned_vs_tuned.csv').set_index('configuration').T")
md("""
## 5. HMM / Viterbi (Section 5)
Decoder parameters tuned on the train out-of-fold probabilities (no student-test involved):
""")
code("""
cfg = yaml.safe_load(open('configs/hmm.yaml'))
cfg
""")
code("""
S = pd.read_csv('results/hmm/hmm_vs_classifier_summary.csv')
S
""")
code("""
D = pd.read_csv('results/hmm/hmm_vs_classifier_per_trip.csv', dtype={'trip_id': str})
D[D.partition == 'student_test'].pivot(index='trip_id', columns='method', values='top1')
""")
code("show('results/figures/hmm_vs_classifier_top1.png')")
code("show('results/figures/hmm_vs_classifier_continuity.png')")
code("show('results/figures/hmm_decoder_tuning.png')")
md("### What kind of errors does the HMM fix?")
code("pd.read_csv('results/error_analysis.csv').set_index('method').T")
md("### Are the remaining continuity breaks real jumps?")
code("pd.read_csv('results/hmm/break_analysis.csv')")
md("""
All remaining breaks are multi-hop: a directed road path of ≤ 300 m joins the two arcs (the vehicle crossed short arcs within
one GPS interval). The decoded route is connected on the road graph; the continuity metric (same or directly adjacent arc) is strict.

## 6. Every model through the final decoder
""")
code("pd.read_csv('results/model_route_metrics.csv').sort_values('hmm_top1', ascending=False)")
md("## 7. End-to-end API latency (same code as `POST /predict`)")
code("""
lat = pd.read_csv('results/api_latency.csv', dtype={'trip_id': str})
print('mean total', round(lat.total_ms.mean()), 'ms; max', lat.total_ms.max(), 'ms')
lat
""")


# --------------------------------------------------------------------------------------------- execution
def show(path):
    """Display a PNG produced by the experiment scripts (stored as an image output)."""
    _IMAGES.append(base64.b64encode(Path(path).read_bytes()).decode())


def run():
    ns = {"show": show}
    nb_cells, count = [], 0
    for kind, src in CELLS:
        if kind == "markdown":
            nb_cells.append({"cell_type": "markdown", "metadata": {}, "source": src.splitlines(True)})
            continue
        count += 1
        _IMAGES.clear()
        buf, outputs = io.StringIO(), []
        lines = src.splitlines()
        body, last = "\n".join(lines[:-1]), lines[-1]
        try:
            with contextlib.redirect_stdout(buf):
                try:                       # value of the last line, like Jupyter
                    compile(last, "<cell>", "eval")
                    exec(body, ns); value = eval(last, ns)
                except SyntaxError:
                    exec(src, ns); value = None
        except Exception:
            value = None
            outputs.append({"output_type": "error", "ename": "Error", "evalue": "see traceback",
                            "traceback": traceback.format_exc().splitlines()})
        if buf.getvalue():
            outputs.insert(0, {"output_type": "stream", "name": "stdout", "text": buf.getvalue().splitlines(True)})
        for img in _IMAGES:
            outputs.append({"output_type": "display_data", "metadata": {}, "data": {"image/png": img, "text/plain": ["<Figure>"]}})
        if value is not None:
            data = {"text/plain": repr(value).splitlines(True)}
            if hasattr(value, "to_html"):
                data["text/html"] = value.to_html().splitlines(True)
            outputs.append({"output_type": "execute_result", "execution_count": count, "metadata": {}, "data": data})
        nb_cells.append({"cell_type": "code", "execution_count": count, "metadata": {}, "outputs": outputs,
                         "source": src.splitlines(True)})
        errors = [o for o in outputs if o["output_type"] == "error"]
        print(f"cell {count:2d}: {'ERROR' if errors else 'ok'}", flush=True)
        if errors:
            print("\n".join(errors[0]["traceback"][-5:]))
    nb = {"cells": nb_cells, "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                                          "language_info": {"name": "python", "version": sys.version.split()[0]}},
          "nbformat": 4, "nbformat_minor": 5}
    out = ROOT / "notebooks" / "02_models.ipynb"
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    print("wrote", out)


_IMAGES = []
if __name__ == "__main__":
    run()
