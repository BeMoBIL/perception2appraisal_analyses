function exportERPs(channels, timePeriod, subjects, input_data_dir, output_data_dir, conds, questions)
    % Process EEG Data
    % Parameters:
    % channels - cell array of channel names
    % timePeriod - 1x2 array specifying the time period in ms [start, end]
    % subjects - array specifying the range of subjects
    % input_data_dir - directory containing the input .set files
    % output_data_dir - directory to save the output .csv files
    % conds - cell array of condition names
    % questions - cell array of question names
    
    % load eeglab
    eeglab;

    % Initialize a cell array to hold all data for each channel
    allData = cell(size(channels, 2), 1);

    % Loop through subjects, conditions
    for subject = subjects
        for cond = 1:length(conds)
            for question = 1:length(questions)
                try
                    % Load data
                    EEG = pop_loadset('filename', ['sub_' num2str(subject) '-epochs-' char(conds(cond)) '' char(questions(question)) '.set'], ...
                                      'filepath', input_data_dir);

                    % Find indexes for our specified time period
                    thisStartIndex = find(EEG.times >= timePeriod(1), 1);
                    [~, closestIndex] = min(abs(EEG.times - timePeriod(2)));
                    thisEndIndex = closestIndex;

                    % Loop through channels
                    for channel = 1:length(channels)
                        % Which channel? Must be written correctly (i.e. case sensitive - check EEG.chanlocs.labels if unsure)
                        channelLabel = channels{channel};

                        % Subset to the desired channel/time/epoch and compute ERP
                        thisDat = EEG.data(strcmpi({EEG.chanlocs.labels}, channelLabel), thisStartIndex:thisEndIndex, :);
                        thisERP = mean(thisDat, 3);

                        % Prepare data for this channel
                        thisSubject = subject;
                        thisConds = conds(cond);
                        thisQuestion = questions(question);
                        timeCol = EEG.times(thisStartIndex:thisEndIndex);

                        % Append to channel-specific data
                        if isempty(allData{channel})
                            allData{channel} = table;
                        end

                        dataRow = table(repmat(thisSubject, length(timeCol), 1), ...
                                        repmat(thisConds, length(timeCol), 1), ...
                                        repmat(thisQuestion, length(timeCol), 1), ...
                                        timeCol', thisERP', ...
                                        'VariableNames', {'Subject', 'Cond', 'Question', 'Time', 'ERP'});

                        allData{channel} = [allData{channel}; dataRow];
                    end
                catch ME
                    if strcmp(ME.identifier, 'MATLAB:load:couldNotReadFile')
                        fprintf('File for subject %d, condition %s, question %s does not exist. Skipping...\n', subject, char(conds(cond)), char(questions(question)));
                        continue;
                    else
                        rethrow(ME);
                    end
                end
            end
        end
    end

    % Export each channel's data to a file
    for channel = 1:length(channels)
        channelLabel = channels{channel};
        fileName = [char(channelLabel) '_' char(num2str(timePeriod(1))) '_' char(num2str(timePeriod(2))) '.csv'];
        writetable(allData{channel}, fullfile(output_data_dir, fileName));
    end
end
