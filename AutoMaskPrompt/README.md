# Overview

Code for AUTOmating MASK generation (with or without TRACKing) via text PROMPT

# Entries

`AutoMaskPromptProduction.py` is a script for mass-producing masks based on image sets, without tracking.

`AutoMaskTrackPrompt.py` is a script for tracking targets based on videos in `./videos`, stored in `./results`

`AutoMaskTrackPrompt2.py` is the MAIN script for the ##DazzleCowIdentifier## project for acquiring the mask dataset from video frames in `./images`, stored in `./results_scattered`

`utils.py` and `./utilities` hold miscellaneous functions.

`./weights` hold the SAMv2 weights which will be downloaded initially for fast inference
