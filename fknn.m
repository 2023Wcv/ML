function [predict_label,test_out] = fknn(sample_in, sample_out, test_in, k, m)
% FKNN Fuzzy k-nearest neighbor classification rule
%
%	Usage:
%	TEST_OUT = FKNNR(SAMPLE_IN, SAMPLE_OUT, TEST_IN, K)
%
%	SAMPLE_IN: Input part of the sample data
%	SAMPLE_OUT: Output part of the sample data
%	TEST_IN: Input part of the test data
%	K: The "k" in "K-NNR"
%	TEST_OUT: Output of the test data according to fuzzy KNNR
%
%	The dimensions of the above matrices is
%
%	SAMPLE_IN: M1xN
%	SAMPLE_OUT: M1xF
%	TEST_IN: M2xN
%	TEST_OUT: M2xF
%
%	where
%	
%	M1 = the no. of sample data
%	N = no. of features
%	F = no. of classes (or categories)
%	M2 = no. of test data
%
%	For more technical details, please refer to the paper:
%
%	J. M. Keller, M. R. Gray, and J. A. Givens, Jr., "A Fuzzy K-Nearest
%	Neighbor Algorithm", IEEE Transactions on Systems, Man, and Cybernetics,
%	Vol. 15, No. 4, pp. 580-585.  
%
%	For selfdemo, type "fknn" with no arguments.
%
%	See also INITFKNN for obtaining a fuzzy version of SAMPLE_OUT.

%	Roger Jang, 990805

if nargin == 0, selfdemo; return; end

if nargin < 5, m = 2; end
if nargin < 4, k = 3; end

sample_n = size(sample_in, 1);
test_n = size(test_in, 1);
feature_n = size(sample_in, 2);
class_n = size(sample_out, 2);

% Euclidean distance matrix
distmat = vecdist(sample_in, test_in);

% knnmat(i,j) = class of i-th nearest point of j-th input vector
% (The size of knnmat is k times test_n.)
[junk, index] = sort(distmat);
% knnmat = reshape(sample_out(index(1:k,:)), k, test_n);

test_out = zeros(test_n, class_n);
for i = 1:test_n,
	neighbor_index = index(1:k, i);
	weight = distmat(neighbor_index, i)'.^(-2/(m-1));
	weight(isinf(weight))=realmax;		% To avoid weight of inf
	test_out(i,:) = weight*sample_out(neighbor_index,:)/(sum(weight));
end
[junk, max_index] = max(test_out');
predict_label = max_index';


% ========== Self demo ==========
function selfdemo

data_n = 50;

data = rand(data_n, 2);
x = data(:, 1);
y = data(:, 2);
class = zeros(data_n, 1);

index = find(y > x);
class(index) = 1;
index = find(y<=x & y>=-x+1);
class(index) = 2;
class(find(class==0)) = 3;

sampledata = [x y class];

%colordef black;
figure;
axis([0 1 0 1]);
box on;
axis equal square

color = {'r', 'g', 'c'};

for i = 1:3,
	index = find(class==i);
	line(x(index), y(index), 'linestyle', 'none', 'marker', '.', ...
		'color', color{i});
end

k = 3;
fuz_sample_out = initfknn(sampledata, k);
index = find(sum(fuz_sample_out.^0.5, 2)~=1); 
%line(x(index), y(index), 'linestyle', 'none', 'marker', 'o', 'color', 'w');

test_in = rand(50, 2);
test_out = fknn([x y], fuz_sample_out, test_in, k);
% Plot test data
line(test_in(:,1), test_in(:,2), 'linestyle', 'none', 'marker', '.', 'color', 'w');

% Plot desired boundaries
line([0 1], [0 1], 'linestyle', ':');
line([0.5 1], [0.5 0], 'linestyle', ':');

legend('Sample data: Class 1', 'Sample data: Class 2',...
	'Sample data: Class 3', 'Test data', -1);

% Plot classification result of the test data
[junk, max_index] = max(test_out');
for i = 1:3,
	index = find(max_index==i);
	line(test_in(index,1), test_in(index,2), 'linestyle', 'none', ...
		'marker', 'o', 'color', color{i});
end

title('The circle color of a sample point shows its predicted class via FKNN.');

%for i = index(:)',
%	text(x(i), y(i), mat2str(fuz_sample_out(i, :), 2));
%end
