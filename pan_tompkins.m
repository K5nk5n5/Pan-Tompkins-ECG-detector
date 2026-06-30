%% Load Data
clear;
close all;

Fs = 1000; % edit if a different Fs is being used

load('4.mat') % edit this line according to your file name
ecg = val(1,:); % edit this line wrt whichever lead you need to use
[locs, mwi, filtered] = detect_r_peaks(ecg, Fs);

%% Raw ECG
figure
plot(ecg)
grid on
title('Raw ECG')

%% Bandpass Filtered ECG
figure
plot(filtered)
grid on
title('Bandpass Filtered ECG (5-15 Hz)')

%% Differentiated ECG
d=diff(filtered);
figure
plot(d)
title('Differentiated plot')

%% Squared ECG
s=d.^2;
figure
plot (s)
title ('Squared plot')

%% Moving Window Integration
figure
plot(mwi)
grid on
title('Moving Window Integration')

%% Final detected R-peaks on raw ECG
figure
plot(ecg)
hold on
plot(locs, ecg(locs), 'ro', 'MarkerSize', 8, 'LineWidth', 1.5)
grid on
title(sprintf('Detected R-peaks (%d beats)', length(locs)))
xlabel('Samples')
ylabel('Amplitude')

%% RR intervals
RR_samples = diff(locs); % gap in samples, to be converted to sec
RR_sec     = RR_samples / Fs;
RR_msec    = RR_sec * 1000;

%% Heart Rate Matrix

HR_inst    = 60 ./ RR_sec;
meanRR_sec = mean(RR_sec);
HR_avg     = 60 / meanRR_sec;

n_intervals = length(RR_sec);

% Build the matrix: one row per RR interval
% Columns: Beat_From | Beat_To | RR_ms | RR_sec | HR_bpm
HR_matrix = zeros(n_intervals, 5);
for k = 1:n_intervals
    HR_matrix(k, :) = [k, k+1, RR_msec(k), RR_sec(k), HR_inst(k)];
end

% Display in matrix form
fprintf('\n----------------- Heart Rate Matrix -----------------\n');
fprintf('%-10s %-10s %-12s %-12s %-12s\n', ...
        'Beat From', 'Beat To', 'RR (ms)', 'RR (sec)', 'HR (bpm)');
fprintf('%s\n', repmat('-', 1, 58));
for k = 1:n_intervals
    fprintf('%-10d %-10d %-12.1f %-12.3f %-12.1f\n', ...
        HR_matrix(k,1), HR_matrix(k,2), ...
        HR_matrix(k,3), HR_matrix(k,4), HR_matrix(k,5));
end
fprintf('%s\n', repmat('-', 1, 58));
fprintf('%-10s %-10s %-12.1f %-12.3f %-12.1f\n', ...
        'MEAN', '', mean(RR_msec), meanRR_sec, HR_avg);
fprintf('---------------X-------------X-------------X-------------\n\n');


%% Instantaneous heart rate across the record
figure
plot(locs(2:end), HR_inst, 'y-o', 'LineWidth', 1.2, 'MarkerFaceColor', 'b')
grid on
title(sprintf('Instantaneous Heart Rate (avg = %.1f bpm)', HR_avg))
xlabel('Sample index of beat')
ylabel('Heart Rate (bpm)')

%% HRV

if length(RR_msec) < 2
    fprintf('Not enough beats to compute HRV (need at least 3).\n');
else

    SDNN = std(RR_msec); % std() uses (N-1) in the denominator by default

    successive_diffs = diff(RR_msec); % diff() subtracts each RR value from the next one.

    RMSSD = sqrt(mean(successive_diffs .^ 2));

    % associated to the parasympathetic nervous system activity
    pNN50 = (sum(abs(successive_diffs) > 50) / length(successive_diffs)) * 100;

    % HRV results in tabular form
    fprintf('\n---------- HRV Summary ----------\n');
    fprintf('SDNN  = %.2f ms\n', SDNN);
    fprintf('RMSSD = %.2f ms\n', RMSSD);
    fprintf('pNN50 = %.1f %%\n', pNN50);
    fprintf('----------------------------------\n\n');

    % HRV bar chart
    figure
    bar([SDNN, RMSSD, pNN50])

    set(gca, 'XTickLabel', {'SDNN (ms)', 'RMSSD (ms)', 'pNN50 (%)'})

    ylabel('Value')
    title('HRV Metrics')
    grid on


    text(1, SDNN  + 0.5, sprintf('%.2f', SDNN),    'HorizontalAlignment', 'center')
    text(2, RMSSD + 0.5, sprintf('%.2f', RMSSD),   'HorizontalAlignment', 'center')
    text(3, pNN50 + 0.5, sprintf('%.1f%%', pNN50), 'HorizontalAlignment', 'center')

end


%% Detect peaks accurately using Pan Tompkins

function [r_locs, mwi, filtered] = detect_r_peaks(ecg, Fs)

% DETECT_R_PEAKS  Pan-Tompkins QRS detector with adaptive thresholding.

    ecg = double(ecg(:)'); %stores the value such that it can hold decimals
                           %arranges data in horizontal format

    %% 1) Bandpass filter (5-15 Hz QRS band)
    nyq = Fs/2;
    [b,a] = butter(3, [5 15]/nyq, 'bandpass');
    filtered = filtfilt(b,a,ecg); %filtfilt fn removes phase shift

    %% 2) Derivative
    deriv = diff(filtered);

    %% 3) Squaring
    sq = deriv.^2;

    %% 4) Moving window integration
    win = max(1, round(0.150*Fs));
    mwi = movmean(sq, win);

    %% Adaptive thresholding
    min_rr = round(0.2*Fs);  % 200 ms is the least period that each peak must have between them

    [~, cand_locs] = findpeaks(mwi, 'MinPeakDistance', min_rr);

    learn_n = min(length(mwi), round(2*Fs));
    seg = mwi(1:learn_n);
    SPKI = 0.25*max(seg);
    NPKI = 0.5*mean(seg);

    THRESHOLD1 = NPKI + 0.25*(SPKI - NPKI);
    THRESHOLD2 = 0.5*THRESHOLD1;

    r_locs = [];
    last_peak_idx = -inf;

    for k = 1:length(cand_locs)
        idx = cand_locs(k);
        peak_val = mwi(idx);

        if peak_val > THRESHOLD1 && (idx - last_peak_idx) >= min_rr
            r_locs(end+1) = idx; %#ok<AGROW>
            SPKI = 0.125*peak_val + 0.875*SPKI;
            last_peak_idx = idx;
        else
            NPKI = 0.125*peak_val + 0.875*NPKI;
        end

        THRESHOLD1 = NPKI + 0.25*(SPKI - NPKI);
        THRESHOLD2 = 0.5*THRESHOLD1;
    end

    %% Back search
    % ensures that no beat was missed due to the qrs complex being below the set threshold
    % finds the avg gap between two beats
    % if the gap between two peaks is more than 1.66*avg, it looks for...
    % peaks within that gap with a lowered threshold
    if length(r_locs) >= 2
        rr_avg = mean(diff(r_locs));
        final_locs = r_locs(1);
        for k = 2:length(r_locs)
            gap = r_locs(k) - final_locs(end);
            if gap > 1.66*rr_avg
                region_mask = (cand_locs > final_locs(end)+min_rr) & ...
                              (cand_locs < r_locs(k));
                region = cand_locs(region_mask);
                if ~isempty(region)
                    [best_val, bi] = max(mwi(region));
                    if best_val > THRESHOLD2
                        final_locs(end+1) = region(bi); %#ok<AGROW>
                    end
                end
            end
            final_locs(end+1) = r_locs(k); %#ok<AGROW>
        end
        r_locs = final_locs;
    end

    %% Find true peak on raw ECG
    % with each stage of the Pan Tompkins algorithm, the peak shifts from...
    % the original position
    % in this stage, within a window of ±80ms, the true local extrema is...
    % located.
    search_radius = max(1, round(0.08*Fs));
    n = length(ecg);
    refined = [];
    for k = 1:length(r_locs)
        idx = r_locs(k);
        lo = max(1, idx-search_radius);
        hi = min(n, idx+search_radius);
        if hi <= lo
            continue
        end
        [~, rel] = max(abs(filtered(lo:hi)));
        refined(end+1) = lo + rel - 1; %#ok<AGROW>
    end

    r_locs = unique(refined);
end