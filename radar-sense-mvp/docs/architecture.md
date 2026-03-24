# Architecture

RADAR-Sense-MVP is organized as a short pipeline:

1. Generate synthetic clean heartbeat modulation signals.
2. Convert them into simple Doppler-like observations.
3. Apply perturbations to create benchmark stress cases.
4. Estimate heart rate from each observation.
5. Compare estimates against objective labels.

The implementation intentionally favors explicitness over physiological realism. Each layer should remain independently testable and fast enough for repeated benchmark runs.
