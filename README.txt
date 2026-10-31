DASE7506 Mini Project 2
Final V8 submission package
Wang Junxin 3036747408
31 October 2026

Result and files

The frozen V8 checkpoint reproduces 395/467 = 84.58244111349036% on the
complete public test bank. The official score is evidence/reproduction/test-score.json.
The editable English report is report/MP2_Report_Wang_Junxin_V8_Final.docx
(8 pages in the verified Word render). This package has not been uploaded or submitted.

code/submission.py is the default V8 entry point. Always pair this code with
this package's checkpoint directory. The separate checkpoint ZIP expands to
model_bundle/; pass that directory with --checkpoint when using it instead.
No retraining is required to reproduce the submitted score.

Directory contents

code/             Standalone inference, fixed training and nested CV code;
                  original course data, parser, prediction runner and scorer.
checkpoint/       Complete frozen V8 assets, including its three base models.
report/           Editable English report.
evidence/         Official scores, resource records, training reproduction,
                  development comparisons, provenance and final designation.
guide/            Original course guide.
licenses/         Retained supplied data license and attribution.
PACKAGE_MANIFEST.json and SHA256SUMS.txt identify the exact release payload.

Inference setup and reproduction on Windows PowerShell

Open a terminal in code/. Use Python 3.12. The following setup uses a new
environment without changing an existing Python installation:

py -3.12 -m venv ../inference_env
$v8Python = '../inference_env/Scripts/python.exe'
& $v8Python -m pip install -r requirements.txt
$env:OMP_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '4'
$env:MKL_NUM_THREADS = '4'
$env:NUMEXPR_NUM_THREADS = '4'
& $v8Python verify_package.py
& $v8Python check_data.py
& $v8Python predict.py --method submission.py --checkpoint ../checkpoint --split test --output runs/reproduced-test-predictions.json
& $v8Python score_predictions.py --split test --predictions runs/reproduced-test-predictions.json --output runs/reproduced-test-score.json

Expected fields: correct=395, n=467, accuracy=84.58244111349036.
Use a fresh output filename when retaining multiple reproduction records.
For validation, replace test with validation in both commands; expected
complete-bank count is 756/798. The hard validation subset is 313/355.

Inference setup on Linux or macOS

From code/:

python3.12 -m venv ../inference_env
../inference_env/bin/python -m pip install -r requirements.txt
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
../inference_env/bin/python verify_package.py
../inference_env/bin/python check_data.py
../inference_env/bin/python predict.py --method submission.py --checkpoint ../checkpoint --split test --output runs/reproduced-test-predictions.json
../inference_env/bin/python score_predictions.py --split test --predictions runs/reproduced-test-predictions.json --output runs/reproduced-test-score.json

Inference uses NumPy 2.5.3 only. Training libraries are not imported by the
prediction path. The code accepts new IDs, bank sizes and candidate orders.
For the instructor's separate bank, use the official --split audit and
--data-dir arguments with the same checkpoint and no retraining.

Training reproduction

Install code/requirements_train_v8.txt in a separate Python 3.12 environment.
Its dependency versions differ from the inference-only environment, so do
not combine both requirement files in one installation. Training is CPU-only
with four threads in the supplied fixed recipe.

From code/, using the training environment's Python:

python -m pip install -r requirements_train_v8.txt
python train_v8.py --checkpoint ../reproduction/model_bundle --record ../reproduction/training_record.json --validate

This rebuilds the base models, five-fold training OOF, PCA32 and fixed
PairLogit head from supplied training data. It does not load the submitted
checkpoint or original development caches. --validate runs only after every
weight is fitted and does not tune or select parameters. Choose a new output
directory; the trainer refuses to overwrite a nonempty checkpoint or record.

The checked local reconstruction took 91.3377 seconds. All numeric checkpoint
arrays and base OOF predictions matched the original exactly, and validation
choices were identical. Evidence is in evidence/retraining/.

To reproduce the nested development evaluation from training data alone:

python cross_validate_v8.py --seed 20261031 --output ../nested_cv_20261031

Expected hard OOF counts: V8=1756/2114 and matched V2=1741/2114. The fresh
checked run took 245.3127 seconds and matched the original inner/outer scores,
head weights and projections exactly. Seed 20261101 is also supported; its
recorded original result is V8=1752 and V2=1739. A separate clean rerun of that
second seed was not needed for this packaging check. All CV outputs must be
kept outside the inference checkpoint.

Neural seed=20261030; neural epochs=5; tree seed=7506; tree rounds=600;
stack seed=20261104; PairLogit depth=4 and iterations=600. Saved original V2
normalizers remain fixed recipe constants. Full configurations, source hashes,
dependencies and train-side fold indices are in the accompanying records.

Resource record

Frozen assets: 1,910,314 uncompressed bytes (1.82 MiB), including every base
model, numeric tree, projection and configuration. No prediction files are
loaded as model assets. The fixed package reproduction on Windows 11,
Intel Core i9-14900HX, Python 3.12.14 and four CPU threads took 3.5853 seconds:
3.1918 for a fresh prediction process plus 0.3935 for a fresh scorer process.
The course reference Xeon Platinum 8457C determines the 30-second limit.
Local timing is not a reference-machine certification.

Development and selection history

V8 was developed through literature/documentation review and training/validation
experiments. It improved the two checked training OOF totals by 15 and 13, and
hard validation by one question relative to V2. Its second seed improved only
two folds, failing the earlier conservative three-positive-fold rule; V2 was
initially retained. Six fixed representatives were subsequently compared on
the public test bank. The final designation of V8 occurred after those scores
were known, with the user citing its existing training/validation evidence.
The full chronology is retained in the report and evidence/final_selection.json.
The public test result is not described as an untouched final-selection holdout.
No per-question test error analysis, weight update or test-driven tuning was
performed after that comparison.

Some immutable historical files say "retain V2" or "test not evaluated";
these describe their earlier development stage. The text status inside the
checkpoint configuration was kept as part of the original pre-test freeze.
The release designation is recorded separately in final_selection.json, and
the numeric checkpoint is unchanged. Original V2 files and four original
archives remain preserved outside this separate V8 package.

Attribution and AI assistance

I reviewed relevant methods and discussed candidate approaches with Codex.
Codex assisted with approach screening, coding, experiments, result checks and
report preparation; I made the final method choice and am responsible for the
submitted work. General method references and official library documentation
appear in the report. External
solution source code and weights are not included in any submitted component.
Course-supplied materials and data attribution
are retained unchanged. No additional pretrained weights or external training
dataset is used.

Submission

Publish the full package contents in a publicly accessible code repository,
including this report and evidence. Provide a publicly accessible link to the
matching checkpoint archive. Enter 84.58244111349036 as the accuracy field,
not 0.845824411 and not the validation accuracy. Identify the final repository
commit/version and matching checkpoint. The MP2 site prepares a GitHub issue;
create that issue on GitHub to actually submit. Deadline: 31 October 2026,
23:59 Hong Kong/Beijing time. Local packaging itself does not submit anything.

Official guide: https://github.com/xudongwu-0/DASE7506-MP2/blob/main/guide/GUIDE.md
Submission site: https://xudongwu-0.github.io/courses/dase7506/mp2/
