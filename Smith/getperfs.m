% Extract the performance traces, learning trials and expert trials of every simulation run from the trial outcomes in
% outcomes_<name>.mat with the method of Smith et al. (2004) and save them to performances.mat. The order of the runs
% must match get_runs() in l5apical/helper.py, which indexes the rows of expert_t and learning_t. The Python port in
% l5apical/smith.py does the same without Matlab (python main.py perf).
nrseeds = 10;
maxres = 1;
corprob = 0.5;
names = ["default", "apical_inhibition", "hebb", "hebb_apical_inhibition", "bcm", "bcm_apical_inhibition", ...
         "calcium", "calcium_apical_inhibition", "fast_default", "fast_apical_inhibition", "fast_hebb", ...
         "fast_hebb_apical_inhibition", "fast_bcm", "fast_bcm_apical_inhibition", "fast_calcium", ...
         "fast_calcium_apical_inhibition"];
nrfiles = length(names);
expert_t = zeros(nrfiles, nrseeds);
learning_t = zeros(nrfiles, nrseeds);
performances = struct();

for f = 1:nrfiles
    load("outcomes_" + names(f) + ".mat")
    n_trials = size(correct_trials, 2);
    perf_traces = zeros(nrseeds, n_trials+1);
    for s = 1:nrseeds
        disp({f, s})
        [t_learn, t_expert, pmid, p05]  = runanalysisv3(correct_trials(s,:), maxres, corprob);
        expert_t(f, s) = t_expert;
        learning_t(f, s) = t_learn;
        perf_traces(s,:) = pmid;
    end
    performances.("perf_" + names(f)) = perf_traces;
end

performances.expert_t = expert_t;
performances.learning_t = learning_t;
save("performances.mat", "-struct", "performances")
