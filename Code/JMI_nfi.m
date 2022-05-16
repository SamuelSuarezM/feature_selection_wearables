function [selectedFeatures] = JMI_nfi(X_data,Y_labels, topK, T)
% Summary 
%    JMI algorithm for feature selection
% Inputs
%    X_data: n x d matrix X, with categorical values for n examples and d features
%    Y_labels: n x 1 vector with the labels
%    topK: Number of features to be selected
%    T: datatypes

numFeatures = size(X_data,2);

score_per_feature = zeros(1,numFeatures,'like', T.score_per_feature);

F = T.score_per_feature.FractionLength;
W = T.score_per_feature.WordLength;
max_value = (2^((W-F)-1)) - 0.01;

for index_feature = 1:numFeatures
    score_per_feature(index_feature) = double(mi_nfi(X_data(:,index_feature),Y_labels, T));
end

selectedFeatures = zeros(1,topK);

[~,selectedFeatures(1)]= max(double(score_per_feature));

not_selected_features = setdiff(1:numFeatures,sort(selectedFeatures));

%%% Implementation of the second step
score_per_feature = zeros(1,numFeatures,'like', T.score_per_feature);
score_per_feature(selectedFeatures(1)) = double(-1); 
 
count = 2;
div = 1;
while count<=topK

    for index_feature_ns = 1:length(not_selected_features)
            score = double(score_per_feature(not_selected_features(index_feature_ns)));
            mi2 = double(mi_nfi([X_data(:,not_selected_features(index_feature_ns)),X_data(:, selectedFeatures(count-1))], Y_labels,T))/div;
            pre_score = score + mi2;
            if pre_score > max_value
                score_per_feature = score_per_feature/2;
                pre_score = pre_score/2;
                div = div*2;
            end
            score_per_feature(not_selected_features(index_feature_ns)) = pre_score;

    end
    [~,selectedFeatures(count)]= max(score_per_feature);
    
   score_per_feature(selectedFeatures(count)) = double(-1);
    not_selected_features = setdiff(1:numFeatures,sort(selectedFeatures));
    count = count+1;
end