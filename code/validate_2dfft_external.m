%% Independent patient-level validation of 2D-FFT feature engineering
% Dataset: Banerjee et al., Zenodo 10.5281/zenodo.4805258 (CC BY 4.0)
% MATLAB R2024b. All spectra are averaged by patient before modelling.

clear; clc; close all;
rng(42, 'twister');
warning('off', 'stats:pca:ColRankDefX');

scriptDir = fileparts(mfilename('fullpath'));
rootDir = fileparts(scriptDir);
dataDir = fullfile(rootDir, 'data', 'zenodo_4805258');
devDir = fullfile(dataDir, 'predictive_extracted', 'Data for predictive model');
testDir = fullfile(dataDir, 'testing_extracted', 'Data for testing the model', ...
    'Blinded Study 10112020', 'CSV Format');
outDir = fullfile(rootDir, 'analysis', 'external_validation');
if ~exist(outDir, 'dir'), mkdir(outDir); end

wnGrid = linspace(1800, 900, 256); % conventional decreasing FTIR direction

severeTestIDs = ["P437","P443","P551","P630","P631","P633","P636", ...
    "P639","P682","P700","P701","P702","P704","P710","P711", ...
    "P712","P713"];
nonsevereTestIDs = ["P679","P436","P438","P439","P440","P441","P448", ...
    "P552","P574","P575","P577","P626","P699"];

[XdevRaw, yDev, idDev, repDev] = loadDevelopment(devDir, wnGrid);
[XtestRaw, yTest, idTest, repTest] = loadIndependentTest(testDir, wnGrid, ...
    severeTestIDs, nonsevereTestIDs);

assert(size(XdevRaw,1) == 130 && sum(yDev==0) == 78 && sum(yDev==1) == 52);
assert(size(XtestRaw,1) == 30 && sum(yTest==0) == 13 && sum(yTest==1) == 17);
assert(all(repDev == 2) && all(repTest == 2), ...
    'Every participant must have exactly two technical replicates.');

% Standard normal variate is applied independently to each patient spectrum.
Xdev = snvRows(XdevRaw);
Xtest = snvRows(XtestRaw);

reps = struct( ...
    'key',   {'Direct','FFT1D','FFT2D_16x16','FFT2D_8x32','FFT2D_32x8','FFT2D_16x16_0255'}, ...
    'label', {'Direct spectrum + PCA','1D-FFT magnitude + PCA','2D-FFT 16x16 magnitude + PCA', ...
              '2D-FFT 8x32 magnitude + PCA','2D-FFT 32x8 magnitude + PCA', ...
              '2D-FFT 16x16 magnitude + PCA with 0-255 scaling'}, ...
    'kind',  {'direct','fft1d','fft2d','fft2d','fft2d','fft2d0255'}, ...
    'rows',  {1,1,16,8,32,16}, ...
    'cols',  {256,256,16,32,8,16});

uniqueRep = struct('key','FFT2D_16x16_unique', ...
    'label','2D-FFT 16x16 unique magnitude + PCA', ...
    'kind','fft2dunique','rows',16,'cols',16);

nModels = numel(reps);
ensembleSize = 25;
allResults = table;
allPredictions = table(idTest, yTest, 'VariableNames', {'PatientID','Reference'});
benchmarkResults = table;
seedResults = table;
modelDetails = struct;

for r = 1:nModels
    [Adev, Atest] = makeRepresentation(Xdev, Xtest, reps(r));
    [Zdev, Ztest, pcaInfo] = trainPCA95(Adev, Atest);
    [prob, memberProb] = trainMLPEnsemble(Zdev, yDev, Ztest, ensembleSize, 1000+r*100);
    pred = double(prob >= 0.5);
    m = binaryMetrics(yTest, pred, prob);

    one = table(string(reps(r).key), string(reps(r).label), pcaInfo.nPC, ...
        m.TN, m.FP, m.FN, m.TP, m.Accuracy, m.Sensitivity, m.Specificity, ...
        m.BalancedAccuracy, m.F1, m.AUROC, m.AUPRC, m.Brier, ...
        m.SensCI(1), m.SensCI(2), m.SpecCI(1), m.SpecCI(2), ...
        'VariableNames', {'Representation','Label','PCs','TN','FP','FN','TP', ...
        'Accuracy','Sensitivity','Specificity','BalancedAccuracy','F1','AUROC', ...
        'AUPRC','Brier','Sensitivity_CI_L','Sensitivity_CI_U', ...
        'Specificity_CI_L','Specificity_CI_U'});
    allResults = [allResults; one]; %#ok<AGROW>
    allPredictions.(reps(r).key + "_Probability") = prob;
    allPredictions.(reps(r).key + "_Prediction") = pred;
    modelDetails.(reps(r).key).pca = pcaInfo;
    modelDetails.(reps(r).key).memberProbabilities = memberProb;
    seedResults = [seedResults; individualSeedMetrics(yTest, memberProb, ...
        string(reps(r).key), string(reps(r).label))]; %#ok<AGROW>

    bench = trainConventionalBenchmarks(Zdev, yDev, Ztest, yTest, string(reps(r).key));
    benchmarkResults = [benchmarkResults; bench]; %#ok<AGROW>
end

% Direct primary contrast: does 2D-FFT add value beyond 1D-FFT?
primaryDelta = pairedBootstrapDeltas(yTest, ...
    allPredictions.FFT1D_Prediction, allPredictions.FFT1D_Probability, ...
    allPredictions.FFT2D_16x16_Prediction, allPredictions.FFT2D_16x16_Probability, ...
    10000, 424242);
paired2Dvs1D = table("2D-FFT 16x16 minus 1D-FFT", ...
    primaryDelta.DeltaBalancedAccuracy, primaryDelta.BalancedCI(1), primaryDelta.BalancedCI(2), ...
    primaryDelta.DeltaAUROC, primaryDelta.AUROCCI(1), primaryDelta.AUROCCI(2), ...
    'VariableNames', {'Contrast','DeltaBalancedAccuracy','DeltaBA_CI_L','DeltaBA_CI_U', ...
    'DeltaAUROC','DeltaAUC_CI_L','DeltaAUC_CI_U'});

% Sensitivity analysis using the unique half of the 2D Fourier plane.
[AuniqueDev, AuniqueTest] = makeRepresentation(Xdev, Xtest, uniqueRep);
[ZuniqueDev, ZuniqueTest, uniquePCA] = trainPCA95(AuniqueDev, AuniqueTest);
[uniqueProb, uniqueMemberProb] = trainMLPEnsemble(ZuniqueDev, yDev, ZuniqueTest, ensembleSize, 9900);
uniquePred = double(uniqueProb >= 0.5);
uniqueMetric = binaryMetrics(yTest, uniquePred, uniqueProb);
nonredundant2D = table(string(uniqueRep.key), string(uniqueRep.label), size(AuniqueDev,2), ...
    uniquePCA.nPC, uniqueMetric.TN, uniqueMetric.FP, uniqueMetric.FN, uniqueMetric.TP, ...
    uniqueMetric.Accuracy, uniqueMetric.Sensitivity, uniqueMetric.Specificity, ...
    uniqueMetric.BalancedAccuracy, uniqueMetric.F1, uniqueMetric.AUROC, ...
    uniqueMetric.AUPRC, uniqueMetric.Brier, ...
    'VariableNames', {'Representation','Label','InputFeatures','PCs','TN','FP','FN','TP', ...
    'Accuracy','Sensitivity','Specificity','BalancedAccuracy','F1','AUROC','AUPRC','Brier'});
seedResults = [seedResults; individualSeedMetrics(yTest, uniqueMemberProb, ...
    string(uniqueRep.key), string(uniqueRep.label))];
uniqueDelta = pairedBootstrapDeltas(yTest, ...
    allPredictions.FFT1D_Prediction, allPredictions.FFT1D_Probability, ...
    uniquePred, uniqueProb, 10000, 424243);
pairedUniqueVs1D = table("Unique 2D-FFT 16x16 minus 1D-FFT", ...
    uniqueDelta.DeltaBalancedAccuracy, uniqueDelta.BalancedCI(1), uniqueDelta.BalancedCI(2), ...
    uniqueDelta.DeltaAUROC, uniqueDelta.AUROCCI(1), uniqueDelta.AUROCCI(2), ...
    'VariableNames', paired2Dvs1D.Properties.VariableNames);

% Matched-complexity analysis: every representation uses the first seven PCs.
matchedPC = 7;
sensitivityReps = [reps uniqueRep];
matchedPCResults = table;
matchedPCPredictions = table(idTest, yTest, 'VariableNames', {'PatientID','Reference'});
for r = 1:numel(sensitivityReps)
    [Adev, Atest] = makeRepresentation(Xdev, Xtest, sensitivityReps(r));
    [Zdev, Ztest, pcaInfo] = trainPCAFixed(Adev, Atest, matchedPC);
    [prob, memberProb] = trainMLPEnsemble(Zdev, yDev, Ztest, ensembleSize, 12000+r*100);
    pred = double(prob >= 0.5);
    m = binaryMetrics(yTest, pred, prob);
    row = table(string(sensitivityReps(r).key), string(sensitivityReps(r).label), ...
        size(Adev,2), pcaInfo.nPC, m.TN, m.FP, m.FN, m.TP, m.Accuracy, ...
        m.Sensitivity, m.Specificity, m.BalancedAccuracy, m.F1, m.AUROC, ...
        m.AUPRC, m.Brier, 'VariableNames', {'Representation','Label','InputFeatures', ...
        'PCs','TN','FP','FN','TP','Accuracy','Sensitivity','Specificity', ...
        'BalancedAccuracy','F1','AUROC','AUPRC','Brier'});
    matchedPCResults = [matchedPCResults; row]; %#ok<AGROW>
    matchedPCPredictions.(sensitivityReps(r).key + "_Probability") = prob;
    matchedPCPredictions.(sensitivityReps(r).key + "_Prediction") = pred;
end
matchedDelta = pairedBootstrapDeltas(yTest, ...
    matchedPCPredictions.FFT1D_Prediction, matchedPCPredictions.FFT1D_Probability, ...
    matchedPCPredictions.FFT2D_16x16_Prediction, matchedPCPredictions.FFT2D_16x16_Probability, ...
    10000, 424244);
pairedMatched2Dvs1D = table("Matched 7-PC 2D-FFT 16x16 minus 1D-FFT", ...
    matchedDelta.DeltaBalancedAccuracy, matchedDelta.BalancedCI(1), matchedDelta.BalancedCI(2), ...
    matchedDelta.DeltaAUROC, matchedDelta.AUROCCI(1), matchedDelta.AUROCCI(2), ...
    'VariableNames', paired2Dvs1D.Properties.VariableNames);

% Paired stratified bootstrap differences relative to the direct-spectrum model.
nBoot = 5000;
directProb = allPredictions.Direct_Probability;
directPred = allPredictions.Direct_Prediction;
deltaRows = table;
for r = 2:nModels
    key = string(reps(r).key);
    p = allPredictions.(key + "_Probability");
    q = allPredictions.(key + "_Prediction");
    d = pairedBootstrapDeltas(yTest, directPred, directProb, q, p, nBoot, 20260+r);
    row = table(key, d.DeltaBalancedAccuracy, d.BalancedCI(1), d.BalancedCI(2), ...
        d.DeltaAUROC, d.AUROCCI(1), d.AUROCCI(2), ...
        'VariableNames', {'Representation','DeltaBalancedAccuracy_vs_Direct', ...
        'DeltaBA_CI_L','DeltaBA_CI_U','DeltaAUROC_vs_Direct','DeltaAUC_CI_L','DeltaAUC_CI_U'});
    deltaRows = [deltaRows; row]; %#ok<AGROW>
end

% Repeated locked-fold internal validation. No hyperparameter or representation
% is selected in this loop; all preprocessing and PCA are refitted per fold.
[cvRows, cvSummary] = repeatedCVRepresentations(Xdev, yDev, reps, 50, 5);

seedSummary = summarizeSeedMetrics(seedResults);

writetable(allResults, fullfile(outDir, 'independent_test_metrics.csv'));
writetable(allPredictions, fullfile(outDir, 'independent_test_predictions.csv'));
writetable(deltaRows, fullfile(outDir, 'paired_bootstrap_differences.csv'));
writetable(paired2Dvs1D, fullfile(outDir, 'paired_bootstrap_2d_vs_1d.csv'));
writetable(pairedMatched2Dvs1D, fullfile(outDir, 'paired_bootstrap_matched_pc_2d_vs_1d.csv'));
writetable(pairedUniqueVs1D, fullfile(outDir, 'paired_bootstrap_unique_2d_vs_1d.csv'));
writetable(benchmarkResults, fullfile(outDir, 'conventional_classifier_benchmarks.csv'));
writetable(cvRows, fullfile(outDir, 'development_repeated_cv.csv'));
writetable(cvSummary, fullfile(outDir, 'development_repeated_cv_summary.csv'));
writetable(seedResults, fullfile(outDir, 'mlp_seed_variability.csv'));
writetable(seedSummary, fullfile(outDir, 'mlp_seed_variability_summary.csv'));
writetable(matchedPCResults, fullfile(outDir, 'matched_pc_sensitivity.csv'));
writetable(matchedPCPredictions, fullfile(outDir, 'matched_pc_predictions.csv'));
writetable(nonredundant2D, fullfile(outDir, 'nonredundant_2d_sensitivity.csv'));
save(fullfile(outDir, 'external_validation_results.mat'), 'allResults', ...
    'allPredictions', 'deltaRows', 'benchmarkResults', 'cvRows', 'cvSummary', ...
    'modelDetails', 'wnGrid', 'idDev', 'idTest', ...
    'yDev', 'yTest', 'repDev', 'repTest', 'reps', 'paired2Dvs1D', ...
    'pairedMatched2Dvs1D', 'pairedUniqueVs1D', 'matchedPCResults', ...
    'matchedPCPredictions', 'nonredundant2D', 'seedResults', 'seedSummary', ...
    'uniqueProb', 'uniquePred', 'uniqueMemberProb', 'uniquePCA', 'uniqueRep');

makeFigures(outDir, allResults, allPredictions, yTest, Xdev, Xtest, wnGrid);
makeSensitivityFigures(outDir, paired2Dvs1D, pairedMatched2Dvs1D, ...
    matchedPCResults, seedResults);

fprintf('\nIndependent test results (n=%d; severe=%d, non-severe=%d)\n', ...
    numel(yTest), sum(yTest==1), sum(yTest==0));
disp(allResults(:, {'Representation','PCs','TN','FP','FN','TP','Accuracy', ...
    'Sensitivity','Specificity','BalancedAccuracy','AUROC','AUPRC','Brier'}));
fprintf('\nPaired bootstrap differences versus direct spectrum\n');
disp(deltaRows);
fprintf('\nPrimary paired 2D-FFT versus 1D-FFT contrast\n');
disp(paired2Dvs1D);
fprintf('\nMatched 7-PC sensitivity analysis\n');
disp(matchedPCResults);
fprintf('\nUnique/nonredundant 2D Fourier sensitivity analysis\n');
disp(nonredundant2D);
fprintf('\nConventional classifier benchmarks\n');
disp(benchmarkResults(:, {'Representation','Classifier','BalancedAccuracy','AUROC','Brier'}));
fprintf('\nRepeated five-fold development validation (ridge logistic; 50 repeats)\n');
disp(cvSummary);
fprintf('\nIndividual MLP seed variability (25 members)\n');
disp(seedSummary);

%% Local functions
function [X, y, ids, replicateCount] = loadDevelopment(devDir, wnGrid)
    files = [dir(fullfile(devDir, 'Non Severe', 'CSV Format', '*.csv')); ...
             dir(fullfile(devDir, 'Severe', 'CSV Format', '*.csv'))];
    idsAll = strings(numel(files),1);
    labels = zeros(numel(files),1);
    spectra = zeros(numel(files), numel(wnGrid));
    for i = 1:numel(files)
        idsAll(i) = extractPatientID(files(i).name);
        labels(i) = double(contains(files(i).folder, [filesep 'Severe' filesep]));
        spectra(i,:) = readAndInterpolate(fullfile(files(i).folder, files(i).name), wnGrid);
    end
    [ids, ~, g] = unique(idsAll, 'stable');
    X = zeros(numel(ids), numel(wnGrid));
    y = zeros(numel(ids),1);
    replicateCount = zeros(numel(ids),1);
    for k = 1:numel(ids)
        idx = (g == k);
        X(k,:) = mean(spectra(idx,:), 1, 'omitnan');
        yy = unique(labels(idx));
        assert(numel(yy)==1, 'Inconsistent labels for %s', ids(k));
        y(k) = yy;
        replicateCount(k) = sum(idx);
    end
end

function [X, y, ids, replicateCount] = loadIndependentTest(testDir, wnGrid, severeIDs, nonsevereIDs)
    files = dir(fullfile(testDir, '*.csv'));
    idsAll = strings(numel(files),1);
    spectra = zeros(numel(files), numel(wnGrid));
    for i = 1:numel(files)
        idsAll(i) = extractPatientID(files(i).name);
        spectra(i,:) = readAndInterpolate(fullfile(files(i).folder, files(i).name), wnGrid);
    end
    [ids, ~, g] = unique(idsAll, 'stable');
    X = zeros(numel(ids), numel(wnGrid));
    y = nan(numel(ids),1);
    replicateCount = zeros(numel(ids),1);
    for k = 1:numel(ids)
        idx = (g == k);
        X(k,:) = mean(spectra(idx,:), 1, 'omitnan');
        replicateCount(k) = sum(idx);
        if any(ids(k) == severeIDs)
            y(k) = 1;
        elseif any(ids(k) == nonsevereIDs)
            y(k) = 0;
        else
            error('No independent-test reference label for %s', ids(k));
        end
    end
    assert(numel(unique([severeIDs, nonsevereIDs])) == 30, 'Test label map must contain 30 unique IDs.');
    assert(all(~isnan(y)), 'Independent-test labels are incomplete.');
end

function id = extractPatientID(filename)
    token = regexp(filename, 'P\d+', 'match', 'once');
    assert(~isempty(token), 'Cannot parse patient ID from %s', filename);
    id = string(token);
end

function x = readAndInterpolate(path, wnGrid)
    t = readtable(path, 'VariableNamingRule', 'preserve');
    wn = double(t{:,1});
    intensity = double(t{:,2});
    ok = isfinite(wn) & isfinite(intensity);
    wn = wn(ok); intensity = intensity(ok);
    [wn, order] = sort(wn, 'ascend');
    intensity = intensity(order);
    x = interp1(wn, intensity, wnGrid, 'pchip');
    assert(all(isfinite(x)), 'Interpolation failed for %s', path);
end

function Z = snvRows(X)
    mu = mean(X, 2);
    sigma = std(X, 0, 2);
    sigma(sigma < eps) = 1;
    Z = (X - mu) ./ sigma;
end

function [Adev, Atest] = makeRepresentation(Xdev, Xtest, rep)
    switch rep.kind
        case 'direct'
            Adev = Xdev;
            Atest = Xtest;
        case 'fft1d'
            Adev = log1p(abs(fft(Xdev, [], 2)));
            Atest = log1p(abs(fft(Xtest, [], 2)));
            Adev = Adev(:,1:129); % nonredundant coefficients for real-valued input
            Atest = Atest(:,1:129);
        case {'fft2d','fft2d0255'}
            if strcmp(rep.kind, 'fft2d0255')
                XdevUse = minmax0255(Xdev);
                XtestUse = minmax0255(Xtest);
            else
                XdevUse = Xdev;
                XtestUse = Xtest;
            end
            Adev = fft2Features(XdevUse, rep.rows, rep.cols);
            Atest = fft2Features(XtestUse, rep.rows, rep.cols);
        case 'fft2dunique'
            Adev = fft2UniqueFeatures(Xdev, rep.rows, rep.cols);
            Atest = fft2UniqueFeatures(Xtest, rep.rows, rep.cols);
        otherwise
            error('Unknown representation: %s', rep.kind);
    end
end

function Y = minmax0255(X)
    lo = min(X, [], 2);
    hi = max(X, [], 2);
    span = hi - lo;
    span(span < eps) = 1;
    Y = 255 * (X - lo) ./ span;
end

function A = fft2Features(X, nr, nc)
    assert(size(X,2) == nr*nc, 'Geometry must use all 256 spectral points.');
    A = zeros(size(X,1), nr*nc);
    for i = 1:size(X,1)
        M = reshape(X(i,:), nr, nc);
        F = fftshift(fft2(M));
        A(i,:) = reshape(log1p(abs(F)), 1, []);
    end
end

function A = fft2UniqueFeatures(X, nr, nc)
    assert(size(X,2) == nr*nc, 'Geometry must use all spectral points.');
    mask = hermitianUniqueMask(nr, nc);
    A = zeros(size(X,1), nnz(mask));
    for i = 1:size(X,1)
        M = reshape(X(i,:), nr, nc);
        F = fft2(M);
        values = log1p(abs(F));
        A(i,:) = values(mask)';
    end
end

function mask = hermitianUniqueMask(nr, nc)
    % For real input, F(u,v)=conj(F(-u mod nr,-v mod nc)). Retain one
    % member of each conjugate pair, including self-conjugate frequencies.
    mask = false(nr,nc);
    for u = 0:nr-1
        for v = 0:nc-1
            partnerU = mod(-u,nr);
            partnerV = mod(-v,nc);
            linearHere = u*nc + v;
            linearPartner = partnerU*nc + partnerV;
            if linearHere <= linearPartner
                mask(u+1,v+1) = true;
            end
        end
    end
end

function [Zdev, Ztest, info] = trainPCA95(Adev, Atest)
    mu = mean(Adev,1);
    sd = std(Adev,0,1);
    sd(sd < 1e-12) = 1;
    D = (Adev - mu) ./ sd;
    T = (Atest - mu) ./ sd;
    [coeff, score, ~, ~, explained] = pca(D, 'Algorithm', 'svd', 'Centered', false);
    nPC = find(cumsum(explained) >= 95, 1, 'first');
    nPC = max(2, min(nPC, size(score,2)));
    Zdev = score(:,1:nPC);
    Ztest = T * coeff(:,1:nPC);
    info = struct('featureMean', mu, 'featureSD', sd, 'coeff', coeff(:,1:nPC), ...
        'explained', explained, 'nPC', nPC, 'cumulativeVariance', sum(explained(1:nPC)));
end

function [Zdev, Ztest, info] = trainPCAFixed(Adev, Atest, nPC)
    mu = mean(Adev,1);
    sd = std(Adev,0,1);
    sd(sd < 1e-12) = 1;
    D = (Adev - mu) ./ sd;
    T = (Atest - mu) ./ sd;
    [coeff, score, ~, ~, explained] = pca(D, 'Algorithm', 'svd', 'Centered', false);
    nPC = min(nPC, size(score,2));
    Zdev = score(:,1:nPC);
    Ztest = T * coeff(:,1:nPC);
    info = struct('featureMean', mu, 'featureSD', sd, 'coeff', coeff(:,1:nPC), ...
        'explained', explained, 'nPC', nPC, 'cumulativeVariance', sum(explained(1:nPC)));
end

function [meanProb, memberProb] = trainMLPEnsemble(X, y, Xtest, nMembers, seedBase)
    targets = zeros(2, numel(y));
    targets(1, y==0) = 1;
    targets(2, y==1) = 1;
    memberProb = zeros(size(Xtest,1), nMembers);
    for j = 1:nMembers
        rng(seedBase+j, 'twister');
        net = patternnet(10, 'trainscg', 'crossentropy');
        net.divideFcn = 'dividetrain';
        net.trainParam.epochs = 500;
        net.trainParam.min_grad = 1e-6;
        net.trainParam.showWindow = false;
        net.trainParam.showCommandLine = false;
        net.performParam.regularization = 0.1;
        net = train(net, X', targets);
        out = net(Xtest');
        memberProb(:,j) = out(2,:)';
    end
    meanProb = mean(memberProb,2);
end

function rows = trainConventionalBenchmarks(X, y, Xtest, yTest, representation)
    rows = table;

    lda = fitcdiscr(X, y, 'DiscrimType', 'linear', 'ClassNames', [0 1]);
    [pred, score] = predict(lda, Xtest);
    rows = [rows; metricRow(representation, "LDA", yTest, pred, score(:,2))]; %#ok<AGROW>

    logit = fitclinear(X, y, 'Learner', 'logistic', 'Regularization', 'ridge', ...
        'Lambda', 1/size(X,1), 'Solver', 'lbfgs', 'ClassNames', [0 1]);
    [pred, score] = predict(logit, Xtest);
    rows = [rows; metricRow(representation, "Ridge logistic", yTest, pred, score(:,2))]; %#ok<AGROW>

    svmLin = fitcsvm(X, y, 'KernelFunction', 'linear', 'BoxConstraint', 1, ...
        'Standardize', false, 'ClassNames', [0 1]);
    svmLin = fitPosterior(svmLin, X, y);
    [pred, score] = predict(svmLin, Xtest);
    rows = [rows; metricRow(representation, "Linear SVM", yTest, pred, score(:,2))]; %#ok<AGROW>

    svmRbf = fitcsvm(X, y, 'KernelFunction', 'rbf', 'KernelScale', 'auto', ...
        'BoxConstraint', 1, 'Standardize', false, 'ClassNames', [0 1]);
    svmRbf = fitPosterior(svmRbf, X, y);
    [pred, score] = predict(svmRbf, Xtest);
    rows = [rows; metricRow(representation, "RBF SVM", yTest, pred, score(:,2))]; %#ok<AGROW>
end

function row = metricRow(representation, classifier, y, pred, prob)
    m = binaryMetrics(y, double(pred), prob);
    row = table(representation, classifier, m.TN, m.FP, m.FN, m.TP, ...
        m.Accuracy, m.Sensitivity, m.Specificity, m.BalancedAccuracy, m.F1, ...
        m.AUROC, m.AUPRC, m.Brier, ...
        'VariableNames', {'Representation','Classifier','TN','FP','FN','TP', ...
        'Accuracy','Sensitivity','Specificity','BalancedAccuracy','F1','AUROC','AUPRC','Brier'});
end

function [rows, summary] = repeatedCVRepresentations(X, y, reps, nRepeats, nFolds)
    rows = table;
    for r = 1:numel(reps)
        for repNo = 1:nRepeats
            rng(30000 + r*100 + repNo, 'twister');
            cv = cvpartition(y, 'KFold', nFolds);
            prob = nan(size(y)); pred = nan(size(y));
            for f = 1:nFolds
                tr = training(cv,f); va = test(cv,f);
                [Atr, Ava] = makeRepresentation(X(tr,:), X(va,:), reps(r));
                [Ztr, Zva] = trainPCA95(Atr, Ava);
                mdl = fitclinear(Ztr, y(tr), 'Learner', 'logistic', ...
                    'Regularization', 'ridge', 'Lambda', 1/sum(tr), ...
                    'Solver', 'lbfgs', 'ClassNames', [0 1]);
                [predFold, scoreFold] = predict(mdl, Zva);
                pred(va) = double(predFold);
                prob(va) = scoreFold(:,2);
            end
            m = binaryMetrics(y, pred, prob);
            row = table(string(reps(r).key), repNo, m.BalancedAccuracy, m.AUROC, ...
                m.AUPRC, m.Brier, 'VariableNames', {'Representation','Repeat', ...
                'BalancedAccuracy','AUROC','AUPRC','Brier'});
            rows = [rows; row]; %#ok<AGROW>
        end
    end
    summary = table;
    for r = 1:numel(reps)
        key = string(reps(r).key);
        z = rows(rows.Representation==key,:);
        row = table(key, median(z.BalancedAccuracy), prctile(z.BalancedAccuracy,25), ...
            prctile(z.BalancedAccuracy,75), min(z.BalancedAccuracy), max(z.BalancedAccuracy), ...
            median(z.AUROC), prctile(z.AUROC,25), prctile(z.AUROC,75), ...
            min(z.AUROC), max(z.AUROC), median(z.Brier), ...
            'VariableNames', {'Representation','MedianBalancedAccuracy','BA_Q1','BA_Q3', ...
            'BA_Min','BA_Max','MedianAUROC','AUC_Q1','AUC_Q3','AUC_Min','AUC_Max','MedianBrier'});
        summary = [summary; row]; %#ok<AGROW>
    end
end

function rows = individualSeedMetrics(y, memberProb, representation, label)
    rows = table;
    for seedIndex = 1:size(memberProb,2)
        prob = memberProb(:,seedIndex);
        pred = double(prob >= 0.5);
        m = binaryMetrics(y, pred, prob);
        row = table(representation, label, seedIndex, m.BalancedAccuracy, ...
            m.AUROC, m.AUPRC, m.Brier, 'VariableNames', {'Representation', ...
            'Label','SeedMember','BalancedAccuracy','AUROC','AUPRC','Brier'});
        rows = [rows; row]; %#ok<AGROW>
    end
end

function summary = summarizeSeedMetrics(rows)
    summary = table;
    keys = unique(rows.Representation,'stable');
    for i = 1:numel(keys)
        z = rows(rows.Representation==keys(i),:);
        one = table(keys(i), z.Label(1), height(z), ...
            median(z.BalancedAccuracy), prctile(z.BalancedAccuracy,25), ...
            prctile(z.BalancedAccuracy,75), min(z.BalancedAccuracy), max(z.BalancedAccuracy), ...
            median(z.AUROC), prctile(z.AUROC,25), prctile(z.AUROC,75), ...
            min(z.AUROC), max(z.AUROC), median(z.AUPRC), ...
            prctile(z.AUPRC,25), prctile(z.AUPRC,75), ...
            median(z.Brier), prctile(z.Brier,25), prctile(z.Brier,75), ...
            'VariableNames', {'Representation','Label','Members','MedianBalancedAccuracy', ...
            'BA_Q1','BA_Q3','BA_Min','BA_Max','MedianAUROC','AUC_Q1','AUC_Q3', ...
            'AUC_Min','AUC_Max','MedianAUPRC','AUPRC_Q1','AUPRC_Q3', ...
            'MedianBrier','Brier_Q1','Brier_Q3'});
        summary = [summary; one]; %#ok<AGROW>
    end
end

function m = binaryMetrics(y, pred, prob)
    C = confusionmat(y, pred, 'Order', [0 1]);
    tn=C(1,1); fp=C(1,2); fn=C(2,1); tp=C(2,2);
    sens = tp/(tp+fn); spec = tn/(tn+fp);
    acc = (tp+tn)/sum(C,'all');
    ppv = tp/max(tp+fp,1);
    f1 = 2*ppv*sens/max(ppv+sens,eps);
    [~,~,~,auc] = perfcurve(y, prob, 1);
    auprc = averagePrecision(y, prob);
    [~, sensCI] = binofit(tp, tp+fn, 0.05);
    [~, specCI] = binofit(tn, tn+fp, 0.05);
    m = struct('TN',tn,'FP',fp,'FN',fn,'TP',tp,'Accuracy',acc, ...
        'Sensitivity',sens,'Specificity',spec,'BalancedAccuracy',(sens+spec)/2, ...
        'F1',f1,'AUROC',auc,'AUPRC',auprc,'Brier',mean((prob-y).^2), ...
        'SensCI',sensCI,'SpecCI',specCI);
end

function ap = averagePrecision(y, score)
    [~, order] = sort(score, 'descend');
    yy = y(order);
    tp = cumsum(yy==1);
    fp = cumsum(yy==0);
    precision = tp ./ max(tp+fp,1);
    ap = sum(precision(yy==1)) / sum(y==1);
end

function d = pairedBootstrapDeltas(y, predA, probA, predB, probB, nBoot, seed)
    rng(seed, 'twister');
    idx0 = find(y==0); idx1 = find(y==1);
    deltaBA = zeros(nBoot,1); deltaAUC = zeros(nBoot,1);
    for b = 1:nBoot
        ix = [idx0(randi(numel(idx0),numel(idx0),1)); idx1(randi(numel(idx1),numel(idx1),1))];
        ma = binaryMetrics(y(ix), predA(ix), probA(ix));
        mb = binaryMetrics(y(ix), predB(ix), probB(ix));
        deltaBA(b) = mb.BalancedAccuracy - ma.BalancedAccuracy;
        deltaAUC(b) = mb.AUROC - ma.AUROC;
    end
    ma = binaryMetrics(y, predA, probA); mb = binaryMetrics(y, predB, probB);
    d = struct('DeltaBalancedAccuracy',mb.BalancedAccuracy-ma.BalancedAccuracy, ...
        'BalancedCI',prctile(deltaBA,[2.5 97.5]), ...
        'DeltaAUROC',mb.AUROC-ma.AUROC, ...
        'AUROCCI',prctile(deltaAUC,[2.5 97.5]));
end

function makeFigures(outDir, results, predictions, y, Xdev, Xtest, wnGrid)
    colors = [0.15 0.35 0.60; 0.10 0.55 0.45; 0.70 0.35 0.20; ...
              0.55 0.30 0.65; 0.30 0.55 0.75; 0.75 0.55 0.15];

    f1 = figure('Color','w','Position',[100 100 1500 760]);
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    nexttile;
    shortNames = ["Direct + PCA","1D-FFT + PCA","2D-FFT 16x16 + PCA","2D-FFT 8x32 + PCA", ...
        "2D-FFT 32x8 + PCA","2D-FFT 16x16 + PCA 0-255"];
    shortLabels = categorical(shortNames, shortNames, 'Ordinal', true);
    bar(shortLabels, ...
        [results.BalancedAccuracy results.AUROC], 'grouped');
    ylim([0 1]); ylabel('Independent-test performance');
    legend({'Balanced accuracy','AUROC'},'Location','northwest');
    title('Held-out performance by representation'); grid on;
    xtickangle(24);
    nexttile;
    hold on;
    rocRows = [1 2 3 6];
    for k = rocRows
        key = results.Representation(k);
        p = predictions.(key + "_Probability");
        [xroc,yroc] = perfcurve(y,p,1);
        plot(xroc,yroc,'LineWidth',2,'Color',colors(k,:));
    end
    plot([0 1],[0 1],'k--','LineWidth',1);
    xlabel('False-positive rate'); ylabel('True-positive rate');
    title('Independent-test ROC curves'); axis square; grid on;
    legend(results.Label(rocRows),'Location','southeast','Interpreter','none');
    exportgraphics(f1, fullfile(outDir,'figure_external_performance.png'),'Resolution',600);
    exportgraphics(f1, fullfile(outDir,'figure_external_performance.pdf'),'ContentType','vector');
    close(f1);

    f2 = figure('Color','w','Position',[100 100 1300 650]);
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    nexttile; hold on;
    plot(wnGrid, mean(Xdev(yFromSize(Xdev,78,52)==0,:),1), 'LineWidth',2, 'Color',[0.10 0.45 0.70]);
    plot(wnGrid, mean(Xdev(yFromSize(Xdev,78,52)==1,:),1), 'LineWidth',2, 'Color',[0.80 0.30 0.20]);
    set(gca,'XDir','reverse'); xlabel('Wavenumber (cm^{-1})'); ylabel('SNV intensity');
    title('Development cohort mean spectra'); legend({'Non-severe','Severe'}); grid on;
    nexttile; hold on;
    plot(wnGrid, mean(Xtest(y==0,:),1), 'LineWidth',2, 'Color',[0.10 0.45 0.70]);
    plot(wnGrid, mean(Xtest(y==1,:),1), 'LineWidth',2, 'Color',[0.80 0.30 0.20]);
    set(gca,'XDir','reverse'); xlabel('Wavenumber (cm^{-1})'); ylabel('SNV intensity');
    title('Independent test mean spectra'); legend({'Non-severe','Severe'}); grid on;
    exportgraphics(f2, fullfile(outDir,'figure_mean_spectra.png'),'Resolution',600);
    exportgraphics(f2, fullfile(outDir,'figure_mean_spectra.pdf'),'ContentType','vector');
    close(f2);

    % Confusion matrices for the central representation comparison.
    f3 = figure('Color','w','Position',[100 100 1300 430]);
    compareRows = [1 2 3];
    tiledlayout(1,3,'TileSpacing','compact','Padding','compact');
    for z = 1:numel(compareRows)
        k = compareRows(z);
        key = results.Representation(k);
        pred = predictions.(key + "_Prediction");
        C = confusionmat(y,pred,'Order',[0 1]);
        nexttile;
        imagesc(C); axis image; colormap(f3, parula); colorbar;
        xticks([1 2]); xticklabels({'Non-severe','Severe'});
        yticks([1 2]); yticklabels({'Non-severe','Severe'});
        xlabel('Predicted'); ylabel('Reference'); title(results.Label(k),'Interpreter','none');
        for ii=1:2
            for jj=1:2
                text(jj,ii,num2str(C(ii,jj)),'HorizontalAlignment','center', ...
                    'FontWeight','bold','FontSize',14,'Color','w');
            end
        end
    end
    exportgraphics(f3, fullfile(outDir,'figure_external_confusion_matrices.png'),'Resolution',600);
    exportgraphics(f3, fullfile(outDir,'figure_external_confusion_matrices.pdf'),'ContentType','vector');
    close(f3);
end

function makeSensitivityFigures(outDir, pairedDefault, pairedMatched, matchedResults, seedRows)
    f1 = figure('Color','w','Position',[100 100 1500 720]);
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

    nexttile; hold on;
    points = [pairedDefault.DeltaBalancedAccuracy, pairedDefault.DeltaAUROC; ...
              pairedMatched.DeltaBalancedAccuracy, pairedMatched.DeltaAUROC];
    lows = [pairedDefault.DeltaBA_CI_L, pairedDefault.DeltaAUC_CI_L; ...
            pairedMatched.DeltaBA_CI_L, pairedMatched.DeltaAUC_CI_L];
    highs = [pairedDefault.DeltaBA_CI_U, pairedDefault.DeltaAUC_CI_U; ...
             pairedMatched.DeltaBA_CI_U, pairedMatched.DeltaAUC_CI_U];
    y = [1 2 4 5];
    values = [points(1,1), points(1,2), points(2,1), points(2,2)];
    lo = [lows(1,1), lows(1,2), lows(2,1), lows(2,2)];
    hi = [highs(1,1), highs(1,2), highs(2,1), highs(2,2)];
    errorbar(values, y, values-lo, hi-values, 'horizontal', 'o', ...
        'LineWidth',1.8,'MarkerFaceColor',[0.15 0.40 0.65],'Color',[0.15 0.40 0.65]);
    xline(0,'k--','LineWidth',1.2);
    yticks(y); yticklabels({'Default: Delta BA','Default: Delta AUROC', ...
        'Matched 7-PC: Delta BA','Matched 7-PC: Delta AUROC'});
    xlabel('2D-FFT 16x16 minus 1D-FFT');
    title('Paired 95% bootstrap intervals');
    grid on; ylim([0 6]);

    nexttile;
    keyOrder = ["Direct","FFT1D","FFT2D_16x16","FFT2D_16x16_unique"];
    keep = ismember(matchedResults.Representation,keyOrder);
    m = matchedResults(keep,:);
    [~,order] = ismember(keyOrder,m.Representation);
    m = m(order,:);
    names = categorical(["Direct + PCA","1D-FFT + PCA","2D-FFT full + PCA", ...
        "2D-FFT unique + PCA"], ["Direct + PCA","1D-FFT + PCA", ...
        "2D-FFT full + PCA","2D-FFT unique + PCA"], 'Ordinal',true);
    bar(names,[m.BalancedAccuracy m.AUROC],'grouped');
    ylim([0 1]); ylabel('Held-out performance');
    legend({'Balanced accuracy','AUROC'},'Location','southoutside','Orientation','horizontal');
    title('Matched 7-PC sensitivity analysis'); grid on; xtickangle(22);
    exportgraphics(f1, fullfile(outDir,'figure_sensitivity_analyses.png'),'Resolution',600);
    exportgraphics(f1, fullfile(outDir,'figure_sensitivity_analyses.pdf'),'ContentType','vector');
    close(f1);

    f2 = figure('Color','w','Position',[100 100 1450 680]);
    tiledlayout(1,2,'TileSpacing','compact','Padding','compact');
    plotKeys = ["Direct","FFT1D","FFT2D_16x16","FFT2D_16x16_unique"];
    short = ["Direct + PCA","1D-FFT + PCA","2D full + PCA","2D unique + PCA"];
    keep = ismember(seedRows.Representation,plotKeys);
    z = seedRows(keep,:);
    catNames = strings(height(z),1);
    for i=1:numel(plotKeys)
        catNames(z.Representation==plotKeys(i)) = short(i);
    end
    nexttile;
    boxchart(categorical(catNames,short,'Ordinal',true),z.BalancedAccuracy, ...
        'BoxFaceColor',[0.20 0.50 0.72]);
    ylim([0 1]); ylabel('Balanced accuracy'); title('Individual MLP members');
    grid on; xtickangle(20);
    nexttile;
    boxchart(categorical(catNames,short,'Ordinal',true),z.AUROC, ...
        'BoxFaceColor',[0.76 0.36 0.18]);
    ylim([0 1]); ylabel('AUROC'); title('Twenty-five seeds per representation');
    grid on; xtickangle(20);
    exportgraphics(f2, fullfile(outDir,'figure_mlp_seed_variability.png'),'Resolution',600);
    exportgraphics(f2, fullfile(outDir,'figure_mlp_seed_variability.pdf'),'ContentType','vector');
    close(f2);
end

function y = yFromSize(X, n0, n1)
    assert(size(X,1)==n0+n1);
    y = [zeros(n0,1); ones(n1,1)];
end
