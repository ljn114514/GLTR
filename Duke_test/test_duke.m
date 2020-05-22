clc;clear all;close all;
info_test = importdata('data/info_test.txt');
info_query = info_test(1:702, :);
info_gallery = info_test(703:3338, :);


name = 'yourPath/fea';
fea = importdata(name);    %3338*dim
test_feature = make_video_fea(fea);  %L2 

query_feature = test_feature(1:702, :);
gallery_feature = test_feature(703:3338, :);

distance = pdist2(query_feature,gallery_feature,'euclidean');

[row, col] = size(query_feature);
for i=1:row
    good_index = intersect(find(info_gallery(:,3) == info_query(i,3)), find(info_gallery(:,4) ~= info_query(i,4)))';
    junk_index = intersect(find(info_gallery(:,3) == info_query(i,3)), find(info_gallery(:,4) == info_query(i,4)));
    [~,  sort_index1] = sort(distance(i,:));    
    [ap(i), CMC(i, :)] = compute_AP(good_index, junk_index, sort_index1);
end
ap = ap';
CMC = mean(CMC(:,:));
map = mean(ap);  
fprintf('mAP = %f, r1 = %f, r5 = %f, r10 = %f, r20 = %f\n', map, CMC(1), CMC(5), CMC(10), CMC(20));