function extractAndSavePeaks(peakLat, peakRange, posORneg, channels, subjects, input_data_dir, output_data_dir, conds, questions)
    % Extract and save peak amplitudes and latencies for specified channels, subjects, conditions, and questions
    %
    % Parameters:
    % peakLat - latency of the peak to search (ms)
    % peakRange - range to search around peakLat [start, end] (ms)
    % posORneg - 'pos' for peak, 'neg' for trough
    % channels - cell array of channel names
    % subjects - array specifying the range of subjects
    % input_data_dir - directory containing the input .set files
    % output_data_dir - directory to save the output .csv files
    % conds - cell array of condition names
    % questions - cell array of question names

    % Load eeglab
    eeglab;

    % Loop through channels, subjects, conditions
    for channel = 1:size(channels, 2)

        % This channel
        channelLabel = channels(channel);

        % Initialize data containers
        subjectCol = [];
        condCol = [];
        questionCol = {};
        peakAmpCol = [];
        peakLatencyCol = [];

        for subject = subjects

            for cond = 1:size(conds, 2)

                for question = 1:size(questions, 2)

                    try
                        % Load data
                        EEG = pop_loadset('filename', ['sub_' num2str(subject) '-epochs-' char(conds(cond)) '' char(questions(question)) '.set'], ...
                                          'filepath', input_data_dir);

                        % Find indexes for the specified time period
                        thisStartIndex = find(EEG.times >= peakRange(1), 1);
                        [~, closestIndex] = min(abs(EEG.times - peakRange(2)));
                        thisEndIndex = closestIndex;

                        % Find index of desired peak 
                        thisPeakIndex = find(EEG.times >= peakLat, 1);
                        thisPeakIndex = thisPeakIndex - thisStartIndex;

                        % Subset to the desired channel/time/epoch and compute ERP
                        thisDat = EEG.data(strcmpi({EEG.chanlocs.labels}, channelLabel), thisStartIndex:thisEndIndex, :);
                        thisERP = mean(thisDat, 3);

                        if strcmp(posORneg, 'pos')
                            % Get peaks
                            [peakValueTMP, closestPeakIndexTMP] = findpeaks(thisERP);
                            [latDiff, whichPeak] = min(abs(closestPeakIndexTMP - thisPeakIndex));
                            peakValue = peakValueTMP(whichPeak);
                            closestPeakIndex = closestPeakIndexTMP(peakValueTMP == peakValue);

                        elseif strcmp(posORneg, 'neg')
                            [peakValueTMP, closestPeakIndexTMP] = findpeaks(-thisERP);
                            [latDiff, whichPeak] = min(abs(closestPeakIndexTMP - thisPeakIndex));
                            peakValue = -peakValueTMP(whichPeak);
                            closestPeakIndex = closestPeakIndexTMP(-peakValueTMP == peakValue);
                        else
                            error('Please define posORneg as either pos (searching for peaks) or neg (searching for troughs)')
                        end

                        if size(peakValue, 2) < 1
                            peakValue = nan;
                        end

                        % Add data to containers
                        thisSubject = subject;
                        thisConds = conds(cond);
                        thisQuestion = questions(question);
                        thisAmp = peakValue;
                        thisLatency = EEG.times(thisStartIndex + closestPeakIndex);

                        if size(peakValueTMP, 2) < 1
                            thisLatency = nan;
                        end

                        subjectCol = [subjectCol; thisSubject];
                        condCol = [condCol; thisConds];
                        questionCol = [questionCol; thisQuestion];
                        peakAmpCol = [peakAmpCol; thisAmp];
                        peakLatencyCol = [peakLatencyCol; thisLatency];

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

        % Put it all together into a table
        thisPeaktable = table;
        thisPeaktable.subject = subjectCol;
        thisPeaktable.cond = condCol;
        thisPeaktable.question = questionCol;
        thisPeaktable.peakAmp = peakAmpCol;
        thisPeaktable.peakLatency = peakLatencyCol;

        % Dynamically make file name
        if strcmp(posORneg, 'pos')
            namer = 'peak_';
        elseif strcmp(posORneg, 'neg')
            namer = 'trough_';
        end

        fileName = [namer char(channelLabel) '_' char(num2str(peakRange(1))) '_' char(num2str(peakLat)) '_' char(num2str(peakRange(2))) '.csv'];
        filePath = fullfile(output_data_dir, fileName);

        % Check if file exists
        if isfile(filePath)
            % Load existing data
            existingTable = readtable(filePath);
            % Append new data
            combinedTable = [existingTable; thisPeaktable];
            % Remove duplicates
            combinedTable = unique(combinedTable, 'rows');
            % Write the combined table to file
            writetable(combinedTable, filePath);
        else
            % Write the new table to file
            writetable(thisPeaktable, filePath);
        end

    end

end
