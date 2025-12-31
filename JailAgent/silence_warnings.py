"""
Global silent configuration module:
- Suppress TensorFlow INFO / WARNING logs
- Disable oneDNN notifications
- Avoid OpenMP conflicts (libiomp vs libomp)
- Prevent sklearn KMeans memory leak warnings
- Suppress most UserWarning / RuntimeWarning
"""

import os
import warnings

# ------------------- Environment variable configuration -------------------

# Suppress TensorFlow logs (0=show all, 1=hide INFO, 2=hide INFO+WARNING, 3=show ERROR only)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Disable oneDNN notifications
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Avoid conflicts between Intel OpenMP and LLVM OpenMP
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Avoid sklearn KMeans memory leak issues on Windows + MKL
os.environ["OMP_NUM_THREADS"] = "1"

# ------------------- warnings configuration -------------------

# Suppress user warnings from Keras / sklearn / numpy
warnings.filterwarnings("ignore", category=UserWarning)
# Suppress RuntimeWarning (e.g., division by zero / overflow)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# print("[silence_warnings] Global silent configuration applied ✅")
