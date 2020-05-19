%clc;clear all;close all;
addpath 'utils/'
load('info/query_IDX.mat');  % load pre-defined query index
track_test = importdata('info/tracks_test_info.mat');
% train, gallery, and query labels
label_gallery = track_test(:, 3);
label_query = label_gallery(query_IDX);
cam_gallery = track_test(:, 4);
cam_query = cam_gallery(query_IDX);

name = 'YourFeature/fea';
box_feature_test = importdata(name); 
box_feature_test = box_feature_test';

video_feat_test = process_box_feat(box_feature_test);

feat_gallery = video_feat_test;
feat_query = video_feat_test(:, query_IDX);

distance = pdist2(feat_gallery',feat_query','euclidean');
[CMC, map, r1_pairwise, ap_pairwise] = evaluation_mars(distance, label_gallery, label_query, cam_gallery, cam_query);
   
fprintf('single query:   mAP = %f, r1 precision = %f\n', map, CMC(1, 1) );