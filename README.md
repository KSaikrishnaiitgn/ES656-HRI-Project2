# ES656-HRI-Project2: ankle agonist–antagonist activation and a passive spring-shoe constraint

Analysis code for **ES 656 Human–Robot Interaction**, Class Project II (IIT Gandhinagar, Fall 2026), by Kalyan Baidya and Kathroju Saikrishna.

- **Part A (agonist and antagonist activation).** Cyclic dorsiflexion–plantarflexion of the right ankle in single-leg stance. It was performed at three speeds, with a stable 1 kg load at the toes, with the same load pulled at random (unstable), and in three limb configurations. Surface EMG was recorded from the tibialis anterior (TA) and medial gastrocnemius (GAS).
- **Part B (interaction study).** Treadmill walking at 3 km/h, barefoot (trials U1, U2) and wearing a passive spring shoe (C1, C2). The shoe's front and back projections are tied to a band below the knee by wire–spring links. The ankle angle comes from side-view video and is combined with TA/GAS EMG and a mechanical model of the device.

The code does six things:
- tracks the sagittal ankle angle from coloured stickers in video;
- processes surface EMG (BioAmp sensors, RP2040, 2000 Hz) with artefact rejection;
- aligns the EMG with the video;
- models the spring shoe (torque–angle law, stiffness, inertia, interaction forces);
- computes workspace and manipulability of a knee–ankle leg model;
- detects agonist/antagonist bursts and co-contraction.

> **Data are not included in this repository.** The raw EMG files, videos and results were submitted with the course reports. To run the code, place them in the folder layout below.

## Expected folder layout
The scripts read and write folders next to this one. Paths are set in `config_paths.py`:
```
HRI_Project2_PartB/
├── 01_raw_data/emg/            Part B raw BioAmp CSVs (Time_s, Tibialis_Raw_V, Gastro_Raw_V, ...)
├── 01_raw_data/videos/         Part B videos Video1-4.mp4 (U1, U2, C1, C2)
├── PartA/Plots and CSV/        Part A EMG CSVs (Varying_Speed, Configuration, Stable_Loading, Unstable_Loading)
├── PartA/experimental videos/  Part A videos
├── 02_code/                    this repository
├── 03_results/                 outputs (kinematics/, emg/, analysis/, partA/, figures/, videos/)
├── 04_report/figures/          Part B report figures
└── 04_report_PartA/figures/    Part A report figures
```

## Scripts

### Part B pipeline
| Step | Script | What it does | Main outputs |
|---|---|---|---|
| 1 | `s1_track_ankle_angle.py` | HSV sticker tracking, shank–sole angle Θ, neutral calibration from foot-flat frames, outlier rejection, 6 Hz zero-phase filter, OpenSim `.mot` export | `03_results/kinematics/` |
| 2 | `s2_emg_processing_alignment.py` | Notch + 40–450 Hz band-pass + rectification + 6 Hz envelope; clipping and low-frequency artefact masks; EMG–video alignment; gait events; per-stride metrics | `03_results/emg/` |
| 3 | `s3_phase_analysis.py` | Gait timing, ankle events, angular velocity, phase-averaged TA/GAS/co-contraction index | `03_results/emg/detailed_phase_summary.txt` |
| 4 | `s4_spring_shoe_model.py` | Spring-shoe model: geometry reconstruction, nonlinear torque–angle law, stiffness regimes, inertia, swing equilibrium, natural frequencies | printed summary |
| 5 | `s5_workspace_manipulability.py` | Knee–ankle leg model: conventional and walking workspace (holding-effort and spring-extension criteria), kinematic and dynamic manipulability | `04_report/figures/`, `03_results/analysis/` |
| 6 | `s6_interaction_dynamics.py` | Model evaluated along the measured strides: wire tensions, device torque, shoe inertial torque, band forces | `04_report/figures/`, `03_results/analysis/` |
| 7 | `s7_report_figures.py` | Remaining Part B report figures; re-checks the headline numbers | `04_report/figures/` |
| 8 | `s8_render_annotated_videos.py` | Annotated per-trial videos and a compilation video | `03_results/videos/` |

`run_pipeline.py` runs steps 2–7 in order; add `--videos` to include step 8.

### Part A
| Script | What it does | Main outputs |
|---|---|---|
| `pa1_emg_analysis.py` | Same EMG processing as Part B; burst detection; movement cycles defined from TA bursts; normalisation to the normal-speed trial; burst timing and magnitude; antagonist co-activation; co-contraction index; all Part A figures | `03_results/partA/`, `04_report_PartA/figures/` |
| `partA_compile_videos.py` | Compiles the Part A experiment videos with title cards | `03_results/videos/PartA_all_trials_compilation.mp4` |

### Data acquisition (`acquisition/`)
These two files ran during both experiments.

| File | What it does |
|---|---|
| `two_r_pi_emg.ino` | Firmware for the Vicharak Shrike Lite (RP2040). It reads TA on GPIO26/ADC0 and gastrocnemius medialis on GPIO27/ADC1 at 12 bit and 2000 Hz per channel (a 500 µs timer), and streams `adc1,adc2` lines over USB serial at 1 Mbaud. |
| `two_emg_record_filtered.py` | Laptop recording script. It records a fixed duration, converts each reading to volts (V = 3.3·ADC/4095), and saves `Time_s` (nominal n/2000), the raw voltages and its own processed signals: a 20–450 Hz band-pass, a 50 Hz notch with Q = 30, rectification and a 100 ms RMS. It also saves a summary plot. The analysis uses only the raw voltages. |

The recording script needs `pyserial` and `scipy`. Set `PORT` and the output file name for each trial.

### Other
- `config_paths.py`: the folder layout used by every script.
- `util_mp4_timestamps.py`: reads the MP4 creation time and duration, used for EMG–video alignment.
- `diagnostics/`: one-off checks (colour thresholds, tracking checks, EMG spectra).
- `legacy/`: superseded scripts, kept for traceability.

## How to run
```bash
pip install -r requirements.txt
python run_pipeline.py          # Part B, steps 2-7
python pa1_emg_analysis.py      # Part A
```
Step 1 needs the pixel coordinates of the four stickers in a seed frame of each video:
```bash
python s1_track_ankle_angle.py ../01_raw_data/videos/Video2.mp4 ../03_results/kinematics U2 "Unconstrained trial 2" 0 x,y x,y x,y x,y
```
The four x,y pairs are the upper pink, lower pink, yellow heel and yellow toe stickers.

## Conventions
- Θ is the shank–sole angle: 90° = neutral, < 90° = dorsiflexion (DF), > 90° = plantarflexion (PF).
- θ = 90° − Θ is the ankle angle used in the models (+ = DF). It equals OpenSim `ankle_angle_r`.
- No maximum voluntary contraction was recorded, so EMG is **not** %MVC. Part B EMG is a percentage of the unconstrained walking reference peak; Part A EMG is a percentage of the normal-speed trial's reference peak.
- The co-contraction index is CCI = 2·Σmin(TA, GAS) / Σ(TA + GAS) × 100 % (Falconer & Winter, 1985).

