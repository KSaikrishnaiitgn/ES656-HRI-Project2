"""
Run the Part B analysis pipeline in order (steps 2-7; optionally 8).

    python run_pipeline.py            # EMG processing -> phase analysis -> model -> workspace /
                                      # manipulability -> interaction forces -> report figures
    python run_pipeline.py --videos   # also re-render the annotated videos (slow)

Step 1 (s1_track_ankle_angle.py) is not run automatically: it needs four seed points per video
(see README.md). Its outputs are already in 03_results/kinematics/.
"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = ['s2_emg_processing_alignment.py', 's3_phase_analysis.py', 's4_spring_shoe_model.py',
         's5_workspace_manipulability.py', 's6_interaction_dynamics.py', 's7_report_figures.py']
if '--videos' in sys.argv:
    STEPS.append('s8_render_annotated_videos.py')

for step in STEPS:
    print(f'\n=== {step} ===', flush=True)
    r = subprocess.run([sys.executable, os.path.join(HERE, step)], cwd=HERE)
    if r.returncode != 0:
        sys.exit(f'{step} failed with exit code {r.returncode}')
print('\nPipeline finished.')
