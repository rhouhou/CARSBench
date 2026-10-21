import numpy as np
import matplotlib.pyplot as plt

from CARSBench.instrument.calibration import (
    apply_calibration_distortion,
    resample_miscalibrated_spectrum,
)

axis = np.linspace(900.0, 1100.0, 2000)

spectrum = np.exp(
    -0.5 * ((axis - 1000.0) / 4.0) ** 2
)

reported_axis = apply_calibration_distortion(
    axis,
    shift=10.0,
    warp=0.0,
)

shifted = resample_miscalibrated_spectrum(
    spectrum=spectrum,
    nominal_axis=axis,
    reported_axis=reported_axis,
)

plt.plot(axis, spectrum, label="No calibration error")
plt.plot(axis, shifted, label="+10 cm-1 calibration error")

plt.xlabel("Raman shift (cm-1)")
plt.ylabel("Intensity")
plt.legend()
plt.show()