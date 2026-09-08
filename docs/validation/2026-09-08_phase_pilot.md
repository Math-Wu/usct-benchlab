# Source-frequency phase-delay pilot: negative result

This experiment used the unchanged 64-channel high-band D510022534 acquisition
at base 78dea025 plus the equivalent CSR execution change. It did not rerun
k-Wave. A single 500 kHz phase difference relative to water was divided by
angular frequency; the supplied pulse delay selected the integer cycle only.
No GT, shared-channel fitting, or validation values determined preprocessing.
All 2,752 valid channels, weights and validation partition were retained.
This is still a geometric ray travel-time approximation, not finite-frequency
scattering inversion. Single-frequency phase delay is not group delay.

The frequency/threshold policy and full original solver configurations were
frozen before the first run. All four reconstructions completed in this CPU
session, and every returned array and stopping history was saved. Scores below
are tissue metrics using the package's common mask and fixed full-GT range.

| Method | Existing RMSE | Phase RMSE (m/s) | Existing SSIM | Phase SSIM | Phase selected/completed | Phase stop |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| CGLS | 27.193209 | 30.256734 | 0.114274 | 0.082443 | 120/120 | max_iterations |
| SIRT | 27.005599 | 29.321818 | 0.121096 | 0.098822 | 200/200 | max_iterations |
| SART | 26.927737 | 29.288975 | 0.122694 | 0.099221 | 100/100 | max_iterations |
| Bent | 27.087771 | 29.526167 | 0.130122 | 0.100046 | 7/12 | validation_plateau |

**Do not replace the default feature with this pilot.** All four tissue scores
worsened. Phase and pulse-delay residuals have different observation definitions
and cannot be ranked as if they were the same data fidelity. No low-band raw
picker was rerun; no low-band phase reconstruction is claimed at this checkpoint.
The utility is opt-in and remains available for reproducible negative controls.

Nine new unit tests cover known delayed pulses with nonzero time origins,
integer-cycle selection, amplitude-unit invariance, channel-local isolation,
invalid/missing data and half-cycle ambiguity. The intended phasor convention is
exp(-i omega t), with spectra formed by dt * sum(p(t) exp(+i omega t)).
