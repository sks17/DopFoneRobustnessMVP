# RADAR-Sense-MVP

RADAR-Sense-MVP is a minimal research benchmark for evaluating robustness of DopFone-like smartphone fetal-heart-rate sensing under realistic signal artifacts. The project combines synthetic clean signal generation, programmatic perturbations, objective labels, and clean-vs-perturbed evaluation.

## MVP scope

- Generate simple synthetic Doppler-like heartbeat signals near an 18 kHz carrier.
- Apply realistic perturbations such as additive noise, signal dropout, and attenuation.
- Estimate fetal heart rate from clean and perturbed signals.
- Report benchmark metrics that compare estimation quality before and after perturbation.

## Design principles

- Small compositional functions with explicit specifications.
- Datatypes with clearly documented representation and invariants.
- Deterministic generation through explicit seeding.
- Fast end-to-end execution suitable for local experimentation.

## Planned package layout

- `configs/`: YAML configuration for generation, perturbation, and testing.
- `docs/`: architecture and specification notes.
- `src/`: implementation modules.
- `tests/`: unit tests with branch and statement coverage intent.
- `scripts/`: command-line entry points for generation and evaluation.
- `slurm/`: simple batch templates for cluster execution.

## Quick start

1. Create a virtual environment.
2. Install `requirements.txt`.
3. Run `pytest`.
4. Use the scripts in `scripts/` to generate data and run the benchmark.

## Accreditations:

@article{gu2025radar,

  title={RADAR: Benchmarking Language Models on Imperfect Tabular Data}, 

  author={Ken Gu and Zhihan Zhang and Kate Lin and Yuwei Zhang and Akshay Paruchuri and Hong Yu and Mehran Kazemi and Kumar Ayush and A. 
  Ali Heydari and Maxwell A. Xu and Girish Narayanswamy and Yun Liu and Ming-Zher Poh and Yuzhe Yang and Mark Malhotra and Shwetak Patel
   and Hamid Palangi and Xuhai Xu and Daniel McDuff and Tim Althoff and Xin Liu},

  year={2025},

  eprint={2506.08249},

  archivePrefix={arXiv},

  primaryClass={cs.DB},
  
  url={https://arxiv.org/abs/2506.08249}, 
}
