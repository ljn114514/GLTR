function video_feat = make_video_fea(fea_all)

video_feat = fea_all;
sum_val = sqrt(sum(video_feat.^2, 2));
for n = 1:size(video_feat, 2)
    video_feat(:, n) = video_feat(:, n)./sum_val;
end