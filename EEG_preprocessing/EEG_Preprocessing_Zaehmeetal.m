eeglab;
ua_config;

%% Importing BIDS files and converting to EEGLAB .set

addpath(genpath("./bids-example-specification-main/")); %adjust path

general = readstruct(fullfile('hardware-specification','general.json'));
eeg = readstruct(fullfile('hardware-specification','eeg.json'));
env = readstruct(fullfile('data-management-specification','bemobil.json'));
subject = readstruct(fullfile('data-management-specification','subject.json'));

subj = [1:100];

% Set up BIDS directory and output folders
subject.bids_target_folder = 'YOUR_BIDS_DIRECTORY_PATH'; % <-- set this to your BIDS root
subject.set_folder = fullfile(subject.bids_target_folder, 'derivatives', env.raw_eeglab_folder{1});

subject.session_names = {'a', 'b'}; % or whatever sessions you have

subject.match_electrodes_channels = {eeg.eeg_chanloc_names};

for s = subj
    subject.subject = s;
    bemobil_bids2set(subject);
end

%% Preprocessing EEG
subjects = [1:100]; % Specify the subject numbers

bemobil_config.channels_to_remove = {'ECG', 'GSR_MR_50_xx'}; % ECG and EDA data were recorded via ExG amp and are thus part of the EEG dataset as "channels". Not used, so drop here.
bemobil_config.merged_filename = 'merged_EEG.set';

for subject = subjects
    force_recompute = 1;
    bemobil_pipeline;
end

%% Event processing: parsing EEG.event

for s = subjects
    EEG = pop_loadset([bemobil_config.study_folder filesep ...
        bemobil_config.single_subject_analysis_folder filesep ...
        'sub-' num2str(s) filesep 'sub-' num2str(s) '_' ...
        bemobil_config.single_subject_cleaned_ICA_filename]);

    % Parse events
    EEG.event = eeglab_parse_key_val(EEG.event, 'type');
end

%% Epoching for ERPs per stim per scale
EEGDir = fullfile(pwd, '5_single-subject-EEG-analysis');
outputepochDir = fullfile(pwd, 'epochs');
eventsFile = fullfile(pwd, 'scripts','eventsStimScale.mat');
epoch_perstim_perscale(EEGDir, outputepochDir, eventsFile, subjects);

%% export ERPs for processing in R per channel 
channels = {'Pz','Oz','POz','O1','O2','CPz','P3','P4'}; %{'Fp1' 'Fz' 'F3' 'F7' 'FT9' 'FC5' 'FC1' 'C3' 'T7' 'TP9' 'CP5' 'CP1' 'Pz' 'P3' 'P7' 'O1' 'Oz' 'O2' 'P4' 'P8' 'TP10' 'CP6' 'CP2' 'Cz' 'C4' 'T8' 'FT10' 'FC6' 'FC2' 'F4' 'F8' 'Fp2' 'AF7' 'AF3' 'AFz' 'F1' 'F5' 'FT7' 'FC3' 'C1' 'C5' 'TP7' 'CP3' 'P1' 'P5' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'P6' 'P2' 'CPz' 'CP4' 'TP8' 'C6' 'C2' 'FC4' 'FT8' 'F6' 'AF8' 'AF4' 'F2' 'FCz' 'Iz'};
timePeriod = [-200, 1000];
input_data_dir = fullfile(pwd, 'epochs');
output_data_dir = fullfile(pwd, 'epochs','erps');
conds = {'HTHB02' 'HTHB04' 'HTHB05' 'HTHB06' 'HTHB08' 'HTHB09' 'HTHB12' 'HTHB14' 'HTHB15' 'HTHB16' 'HTHB18' 'HTHB19' 'HTHB21' 'HTHB22' 'HTLB03' 'HTLB04' 'HTLB05' 'HTLB06' 'HTLB09' 'HTLB11' 'HTLB12' 'HTLB13' 'HTLB15' 'HTLB16' 'HTLB17' 'HTLB21' 'HTLB22' 'HTLB23' 'LTHB03' 'LTHB04' 'LTHB05' 'LTHB07' 'LTHB08' 'LTHB09' 'LTHB10' 'LTHB11' 'LTHB14' 'LTHB16' 'LTHB18' 'LTHB19' 'LTHB20' 'LTHB25' 'LTLB01' 'LTLB02' 'LTLB03' 'LTLB04' 'LTLB06' 'LTLB07' 'LTLB11' 'LTLB12' 'LTLB14' 'LTLB15' 'LTLB17' 'LTLB19' 'LTLB20' 'LTLB22'};
questions = {'SAM-arousal' 'SAM-dominance' 'SAM-valence' 'Likert-Faszination' 'Likert-Heimeligkeit' 'Likert-Sch_nheit' 'Likert-Sicherheit' 'Likert-Stress' 'Likert-Offenheit'};

exportERPs(channels, timePeriod, subjects, input_data_dir, output_data_dir, conds, questions);


%% Extract P1 and N1 Peaks from identified range

%P1
channels = {'Oz','POz','O1','O2'}; %{'Fp1' 'Fz' 'F3' 'F7' 'FT9' 'FC5' 'FC1' 'C3' 'T7' 'TP9' 'CP5' 'CP1' 'Pz' 'P3' 'P7' 'O1' 'Oz' 'O2' 'P4' 'P8' 'TP10' 'CP6' 'CP2' 'Cz' 'C4' 'T8' 'FT10' 'FC6' 'FC2' 'F4' 'F8' 'Fp2' 'AF7' 'AF3' 'AFz' 'F1' 'F5' 'FT7' 'FC3' 'C1' 'C5' 'TP7' 'CP3' 'P1' 'P5' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'P6' 'P2' 'CPz' 'CP4' 'TP8' 'C6' 'C2' 'FC4' 'FT8' 'F6' 'AF8' 'AF4' 'F2' 'FCz' 'Iz'};
peakLats = [165] % Latency of the peak to search (ms) plus 60ms set-up latency
peakRange = [peakLats - 20, peakLats + 20]; % Range to search around peakLat (ms)
input_data_dir = fullfile(pwd, 'epochs','erps'); % Input data directory
output_data_dir = fullfile(pwd, 'epochs','erps','peaks'); % Output data directory
posORneg = 'pos';

extractAndSavePeaks(peakLats, peakRange, posORneg, channels, subjects, ...
    input_data_dir, output_data_dir, conds, questions);

%N1
channels = {'Oz','POz','O1','O2'}; %{'Fp1' 'Fz' 'F3' 'F7' 'FT9' 'FC5' 'FC1' 'C3' 'T7' 'TP9' 'CP5' 'CP1' 'Pz' 'P3' 'P7' 'O1' 'Oz' 'O2' 'P4' 'P8' 'TP10' 'CP6' 'CP2' 'Cz' 'C4' 'T8' 'FT10' 'FC6' 'FC2' 'F4' 'F8' 'Fp2' 'AF7' 'AF3' 'AFz' 'F1' 'F5' 'FT7' 'FC3' 'C1' 'C5' 'TP7' 'CP3' 'P1' 'P5' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'P6' 'P2' 'CPz' 'CP4' 'TP8' 'C6' 'C2' 'FC4' 'FT8' 'F6' 'AF8' 'AF4' 'F2' 'FCz' 'Iz'};
peakLats = [212] % Latency of the peak to search (ms) plus 60ms set-up latency
peakRange = [peakLats - 20, peakLats + 20]; % Range to search around peakLat (ms)
input_data_dir = fullfile(pwd, 'epochs','erps'); % Input data directory
output_data_dir = fullfile(pwd, 'epochs','erps','peaks'); % Output data directory
posORneg = 'neg';

extractAndSavePeaks(peakLats, peakRange, posORneg, channels, subjects, ...
    input_data_dir, output_data_dir, conds, questions);

%% Extract P3 and LPP Peaks from identified range

%P3
channels = {'CPz','Pz','P3','P4'}; %{'Fp1' 'Fz' 'F3' 'F7' 'FT9' 'FC5' 'FC1' 'C3' 'T7' 'TP9' 'CP5' 'CP1' 'Pz' 'P3' 'P7' 'O1' 'Oz' 'O2' 'P4' 'P8' 'TP10' 'CP6' 'CP2' 'Cz' 'C4' 'T8' 'FT10' 'FC6' 'FC2' 'F4' 'F8' 'Fp2' 'AF7' 'AF3' 'AFz' 'F1' 'F5' 'FT7' 'FC3' 'C1' 'C5' 'TP7' 'CP3' 'P1' 'P5' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'P6' 'P2' 'CPz' 'CP4' 'TP8' 'C6' 'C2' 'FC4' 'FT8' 'F6' 'AF8' 'AF4' 'F2' 'FCz' 'Iz'};
timeRange = [399,449] % Timerange of the peak (ms) for averaging plus 60ms set-up latency
input_data_dir = fullfile(pwd, 'epochs','erps'); % Input data directory
output_data_dir = fullfile(pwd, 'epochs','erps','peaks'); % Output data directory
posORneg = 'pos';

extractAndSaveAverages(timeRange, channels, subject, ...
    input_data_dir, output_data_dir, conds, questions);

%LPP
channels = {'CPz','Pz','P3','P4'}; %{'Fp1' 'Fz' 'F3' 'F7' 'FT9' 'FC5' 'FC1' 'C3' 'T7' 'TP9' 'CP5' 'CP1' 'Pz' 'P3' 'P7' 'O1' 'Oz' 'O2' 'P4' 'P8' 'TP10' 'CP6' 'CP2' 'Cz' 'C4' 'T8' 'FT10' 'FC6' 'FC2' 'F4' 'F8' 'Fp2' 'AF7' 'AF3' 'AFz' 'F1' 'F5' 'FT7' 'FC3' 'C1' 'C5' 'TP7' 'CP3' 'P1' 'P5' 'PO7' 'PO3' 'POz' 'PO4' 'PO8' 'P6' 'P2' 'CPz' 'CP4' 'TP8' 'C6' 'C2' 'FC4' 'FT8' 'F6' 'AF8' 'AF4' 'F2' 'FCz' 'Iz'};
timeRange = [445,545] % Timerange of the peak (ms) for averaging plus 60ms set-up latency
input_data_dir = fullfile(pwd, 'epochs','erps'); % Input data directory
output_data_dir = fullfile(pwd, 'epochs','erps','peaks'); % Output data directory
posORneg = 'pos';

extractAndSaveAverages(timeRange, channels, subject, ...
    input_data_dir, output_data_dir, conds, questions);






