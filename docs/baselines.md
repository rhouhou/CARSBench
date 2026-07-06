# Baseline benchmark

CARSBench includes a lightweight baseline benchmark script for evaluating simple non-learning methods across all benchmark domains.

The purpose of this benchmark is not to provide a strong Raman-retrieval model. Instead, it provides a simple sanity check that the dataset generation, Raman targets, metrics, and domain-level evaluation workflow are working correctly.

---

## Script

Run:

```bash
python scripts/12_run_baseline_benchmark.py
```

The script generates a small synthetic dataset for each benchmark domain and evaluates simple baselines against the Raman-equivalent target signal.

---

## Output

The script saves results to:

```text
results/benchmark/baseline_results.csv
```

Each row contains one baseline method evaluated on one domain.

Typical columns include:

| Column                | Description                          |
| --------------------- | ------------------------------------ |
| `domain`              | Benchmark domain name                |
| `baseline`            | Baseline method name                 |
| `num_samples`         | Number of generated samples          |
| `seed`                | Random seed used for generation      |
| `rmse_mean`           | Mean root mean squared error         |
| `rmse_std`            | Standard deviation of RMSE           |
| `mae_mean`            | Mean absolute error                  |
| `mae_std`             | Standard deviation of MAE            |
| `spectral_angle_mean` | Mean spectral angle                  |
| `spectral_angle_std`  | Standard deviation of spectral angle |

---

## Included baselines

The current script includes simple baselines such as:

| Baseline           | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| `zero`             | Predicts an all-zero Raman target                                  |
| `normalized_input` | Uses the normalized CARS/BCARS spectrum as a naive target estimate |
| `smoothed_input`   | Uses a smoothed and normalized version of the input spectrum       |

These are intentionally simple. They provide reference values for checking whether future methods improve over trivial behavior.

---

## Metrics

### RMSE

Root mean squared error measures the average squared difference between the predicted signal and the Raman-equivalent target.

Lower is better.

### MAE

Mean absolute error measures the average absolute difference between prediction and target.

Lower is better.

### Spectral angle

Spectral angle measures shape similarity between two spectra.

Lower values indicate more similar spectral shapes.

---

## Recommended use

Use this benchmark after making changes to the simulator, domain presets, or dataset format.

It can help check whether:

* all domains can still be generated
* Raman-equivalent targets are available
* metrics can be computed without errors
* domain-level differences are measurable
* future retrieval methods improve over simple baselines

---

## Notes

The baseline benchmark is a starting point.

Future benchmark methods may include:

* Kramers-Kronig phase retrieval
* MEM-based retrieval
* classical preprocessing pipelines
* neural-network baselines
* domain-generalization models
* integration with `prCARS`
* validation with `CARSGuard`
