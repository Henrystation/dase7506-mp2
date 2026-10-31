# Run before launching Python to keep all numeric libraries within four threads.
$env:OMP_NUM_THREADS='4'
$env:OPENBLAS_NUM_THREADS='4'
$env:MKL_NUM_THREADS='4'
$env:NUMEXPR_NUM_THREADS='4'
# Inference: python -m pip install -r requirements.txt
# Separate training environment: python -m pip install -r requirements_train_v8.txt
