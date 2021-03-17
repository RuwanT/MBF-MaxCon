function matches = generate_pseudoconvex_data(img1, img2)

if ischar(img1)
    imargb = im2double(imread(img1));
    imcrgb = im2double(imread(img2));
else
    imargb = im2double(img1);
    imcrgb = im2double(img2);
end

matches.im1 = imargb; 
matches.im2 = imcrgb; 

[fa,da] = vl_sift(im2single(rgb2gray(imargb))) ; 
[fc,dc] = vl_sift(im2single(rgb2gray(imcrgb))) ;

xa = fa(1, :); 
ya = fa(2, :); 
xc = fc(1, :); 
yc = fc(2, :); 

% %
% Compute tentative matches between image 1 (a) and 2 (b) by matching local features
% %
 
[match, ~] = vl_ubcmatch(da, dc, 2.222); 
xat       = xa(match(1, :));
yat       = ya(match(1, :));
xct       = xc(match(2, :));
yct       = yc(match(2, :));

% Pad data with homogeneous scale factor of 1
matches.X1 = [[xat; yat]; ones(1,numel(xat))];
matches.X2 = [[xct; yct]; ones(1,numel(xct))];        

X = unique([matches.X1', matches.X2'], 'rows');
matches.X1 = X(:, 1:3)'; 
matches.X2 = X(:, 4:6)'; 
 
matches.nbpoints = size(matches.X1, 2); 

% X = [ matches.X1; matches.X2 ]; 

[matches.x1, matches.T1] = normalise2dpts(matches.X1);
[matches.x2, matches.T2] = normalise2dpts(matches.X2);

matches.u1 = [matches.x1; zeros(6, matches.nbpoints); matches.x1];
matches.u2 = matches.x2(1:2, :); 
matches.u2 = repmat(matches.u2(:), 1, 3).*reshape([matches.x1;matches.x1], 3, 2*matches.nbpoints)'; 
matches.A = [reshape(matches.u1, [6, 2*matches.nbpoints])', -matches.u2]; 
matches.b = zeros(2, matches.nbpoints); 
matches.c = [zeros(6, matches.nbpoints); matches.x1]; 
matches.d = zeros(1, matches.nbpoints); 