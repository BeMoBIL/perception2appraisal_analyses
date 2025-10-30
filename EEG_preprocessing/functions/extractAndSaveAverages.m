function extractAndSaveAverages(timeRange, channels, subjects, input_data_dir, output_data_dir, conds, questions)
    % Extract and save average amplitudes for specified channels, subjects, conditions, and questions
    %
    % Parameters:
    % timeRange - range to average [start, end] (ms)
    % channels - cell array of channel names
    % subjects - array specifying the range of subjects
    % input_data_dir - directory containing the input .set files
    % output_data_dir - directory to save the output .csv files
    % conds - cell array of condition names
    % questions - cell array of question names

        eeglab;

    % Prepare containers for all results
    allResults = [];

    for subject = subjects
        for cond = 1:numel(conds)
            for question = 1:numel(questions)
                try
                    % Load data ONCE per file
                    EEG = pop_loadset('filename', ['sub_' num2str(subject) char(questions(question)) '-epochs-' char(conds(cond)) '.set'], ...
                                      'filepath', input_data_dir);

                    % Find indexes for the specified time period
                    thisStartIndex = find(EEG.times >= timeRange(1), 1);
                    [~, closestIndex] = min(abs(EEG.times - timeRange(2)));
                    thisEndIndex = closestIndex;

                    % For each channel, extract and store result
                    for channel = 1:numel(channels)
                        channelLabel = channels{channel};
                        chanIdx = find(strcmpi({EEG.chanlocs.labels}, channelLabel));
                        if isempty(chanIdx)
                            warning('Channel %s not found for subject %d, condition %s, question %s. Skipping...', ...
                                channelLabel, subject, char(conds(cond)), char(questions(question)));
                            continue;
                        end
                        thisDat = EEG.data(chanIdx, thisStartIndex:thisEndIndex, :);
                        thisERP = mean(thisDat, 3);
                        avgValue = mean(thisERP);

                        % Store result as a row
                        allResults = [allResults; {subject, conds{cond}, questions{question}, channelLabel, avgValue}];
                    end

                catch ME
                    if strcmp(ME.identifier, 'MATLAB:load:couldNotReadFile')
                        fprintf('File for subject %d, condition %s, question %s does not exist. Skipping...\n', ...
                            subject, char(conds(cond)), char(questions(question)));
                        continue;
                    else
                        rethrow(ME);
                    end
                end
            end
        end
    end

    % Convert to table
    if ~isempty(allResults)
        resultTable = cell2table(allResults, ...
            'VariableNames', {'subject', 'cond', 'question', 'channel', 'avgAmp'});

        % For each channel, save a separate file
        for channel = 1:numel(channels)
            channelLabel = channels{channel};
            chanRows = strcmp(resultTable.channel, channelLabel);
            thisChanTable = resultTable(chanRows, {'subject', 'cond', 'question', 'avgAmp'});

            fileName = ['avg_' channelLabel '_' char(num2str(timeRange(1))) '_' char(num2str(timeRange(2))) '.csv'];
            filePath = fullfile(output_data_dir, fileName);

            % Remove duplicates
            if isfile(filePath)
                existingTable = readtable(filePath);
                combinedTable = [existingTable; thisChanTable];
                % Ensure 'question' is cell array of char vectors
                if iscell(combinedTable.question)
                    combinedTable.question = cellfun(@char, combinedTable.question, 'UniformOutput', false);
                end
                combinedTable = unique(combinedTable, 'rows');
                writetable(combinedTable, filePath);
            else
                writetable(thisChanTable, filePath);
            end
        end
    end
end