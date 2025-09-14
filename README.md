# TimeSeriesImputation

This repository contains a collection of experimental and production-ready scripts for time series imputation, covering different data types (univariate, multivariate) and tokenization strategies (multi-token, 2D tokens, hybrid). Each script reflects a stage of development or a specific methodological focus.

## Scripts Overview

- **`run_multi.py`**  
  *Multivariate*  
  Uses **multi-token shapes** for handling multivariate time series data.

- **`run_multiscale.py`**  
  *Univariate*  
  Focused on **single-variable time series imputation** with a multiscale perspective.

- **`run_multivariate_2d.py`**  
  *Multivariate*  
  Implements a **2D tokenization strategy** for multivariate inputs.

- **`run_multivariate.py`**  
  *Multivariate (early-stage script)*  
  Contains multiple early methods, including **mixed token shapes** (referenced in the 2025-05-13 slide deck).

- **`run_multivariate_hybrid.py`**  
  *Multivariate (early-stage script)*  
  Explores a **hybrid tokenization approach**, as introduced in the 2025-06-03 slide deck.
