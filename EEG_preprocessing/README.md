# EEG Preprocessing Pipeline for Urban Appraisal Study

This repository contains MATLAB scripts for preprocessing and analyzing EEG data from the Urban Appraisal study, which investigates neural responses to urban environment stimuli using event-related potentials (ERPs).

## Overview

The pipeline processes raw EEG data (from BIDS format; to be found in OpenNeuro repository number ds006850) through several stages: Preprocessing using BeMoBIL Pipeline, epoching, and ERP extraction. The output can then be further analyzed using the statistical analysis scripts in this repository.

## Directory Structure

```
EEG_preprocessing/
├── bemobil_pipeline.m              # Core preprocessing pipeline using BeMoBIL Pipeline functions
├── EEG_Preprocessing_Zaehmeetal.m  # Main orchestration script
├── config/
│   ├── ua_config.m                 # Configuration parameters
│   ├── eventsStimScale.mat         # Event definitions
│   ├── data-management-specifications/
│   │   ├── bemobil.json            # BeMoBIL configuration for BIDS import
│   │   └── subject.json            # Subject specifications for BIDS import
│   └── hardware-specifications/
│       ├── eeg.json                # EEG hardware specifications for BIDS import
│       └── general.json            # General hardware specifications for BIDS import
└── functions/
    ├── eeglab_parse_key_val.m      # Event parsing utility
    ├── exportERPs.m                # ERP data export
    ├── extractAndSaveAverages.m    # Average amplitude extraction
    └── extractAndSavePeaks.m       # Peak detection and extraction
```

## Prerequisites

### Required Software
- MATLAB (R2019b or later recommended)
- [EEGLAB](https://sccn.ucsd.edu/eeglab/) toolbox
- [BeMoBIL Pipeline](https://github.com/BeMoBIL/bemobil-pipeline) for preprocessing functions

## Configuration

All preprocessing parameters are defined in `config/ua_config.m`.

## To Run

To run preprocessing, use `EEG_Preprocessing_Zaehmeetal.m` script and adjust necessary paths. 

---
*Last updated: October 2025*
