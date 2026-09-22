# Metrics

For panel `p`, let `V_hat` be the mean predicted Target score and `V` the mean gold Target score.

| Metric | Definition | Use |
|---|---|---|
| SAE | `abs(V_hat - V)` | system-mean error on the native scale |
| nSAE | `SAE / (native_max - native_min)` | comparable system-mean error across score scales |
| Macro-nSAE | equal-weight mean of panel nSAE | formal 12-panel aggregate |
| nMAE | mean row absolute error after scale normalization | row-level prediction quality |
| nRMSE | row RMSE after scale normalization | row-level error with larger-error penalty |

SAE/nSAE and MAE/nMAE are not interchangeable. Row errors may cancel in the system mean, and a method can improve nMAE while worsening nSAE.

