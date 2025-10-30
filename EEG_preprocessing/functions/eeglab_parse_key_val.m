function [events] = eeglab_parse_key_val(eeglab_event_struct, event_field)
% CPS_PARSE_EVENTS parses an eeglab event struct by splitting key:value pairs
% seprated by ';', such as in eeglab event sruct. 
% If no separator is found in the event, leave it as is

    events = eeglab_event_struct;
    
    for i = 1:numel(events)    
        current_event = cellstr(strsplit(events(i).(event_field), ';'));

        if numel(current_event) == 1
            try
                key_val = cellstr(strsplit(current_event{1}, ':'));
                events(i).(key_val{1}) = key_val{2};
            end
        end

        if numel(current_event) > 1
            for j=1:length(current_event)
                key_val = cellstr(strsplit(current_event{j}, ':'));

                for j = 1:numel(key_val)
                    key_val{j} = erase(key_val{j},'"');
                end

                % remove whitespaces
                events(i).(strtrim(key_val{1})) = strtrim(key_val{2});
            end
        end

    end
end

