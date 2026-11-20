"""Folder layout of the Part B project. Every script imports its folder paths from here.

HRI_Project2_PartB/
    00_docs/        brief and master notes
    01_raw_data/    emg/ (BioAmp CSV + acquisition plots), videos/ (Video1-4.mp4)
    02_code/        analysis pipeline (this folder; GitHub-ready); diagnostics/ and legacy/ subfolders
    03_results/     kinematics/, emg/, figures/, videos/ produced by the pipeline
    04_report/      current LaTeX report (main.tex, chapters/, figures/, main.pdf)
    05_archive/     earlier report drafts and superseded outputs
"""
import os

SCRIPTS = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPTS)
DOCS = os.path.join(ROOT, '00_docs')
RAW_EMG = os.path.join(ROOT, '01_raw_data', 'emg')
RAW_VIDEOS = os.path.join(ROOT, '01_raw_data', 'videos')
RESULTS = os.path.join(ROOT, '03_results')
RES_KIN = os.path.join(RESULTS, 'kinematics')
RES_EMG = os.path.join(RESULTS, 'emg')
RES_FIG = os.path.join(RESULTS, 'figures')
RES_VID = os.path.join(RESULTS, 'videos')
RES_ANALYSIS = os.path.join(RESULTS, 'analysis')
REPORT = os.path.join(ROOT, '04_report')
REPORT_FIG = os.path.join(REPORT, 'figures')
ARCHIVE = os.path.join(ROOT, '05_archive')
OLD_REPORT_FIG = os.path.join(ARCHIVE, 'old_report_LaTeX', 'figures')

# ---- Part A (agonist-antagonist activation, ankle; recorded 15 Nov 2026) ----
PA_DIR = os.path.join(ROOT, 'PartA')
PA_EMG = os.path.join(PA_DIR, 'Plots and CSV')
PA_VID = os.path.join(PA_DIR, 'experimental videos')
PA_RES = os.path.join(RESULTS, 'partA')
PA_REPORT = os.path.join(ROOT, '04_report_PartA')
PA_REPORT_FIG = os.path.join(PA_REPORT, 'figures')
