# Source before launching Python; use a separate environment for training.
export OMP_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
# python -m pip install -r requirements.txt
# Training only: python -m pip install -r requirements_train_v8.txt
