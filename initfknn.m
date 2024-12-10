function fuz_class = initfknn(sampledata, k)
%INITfknn Initialize fuzzy membership grades of sample output for fuzzy KNN.
%
%	Usage:
%	FUZ_CLASS = INITFKNN(SAMPLE_DATA, K)
%
%	SAMPLE_DATA: Sample data matrix, where each row is a sample data, with
%		the last column being the class information.  Note that the
%		elements in the last column should take values from 1 to F,
%		where F is the no. of classes (or categories).
%	K: The no. of neighbors used in estimating the fuzzy membership grades.
%	FUZ_CLASS: The modified fuzzy membership grades of each sample data.
%
%	The dimensions of the above matrices is
%
%	SAMPLE_DATA: Mx(N+1)
%	K: 1x1
%	FUZ_CLASS: MxF
%
%	where
%	
%	M = the no. of sample data
%	N = no. of features
%	F = no. of classes (or categories)
%
%	For more technical details, please refer to the paper:
%
%	J. M. Keller, M. R. Gray, and J. A. Givens, Jr., "A Fuzzy K-Nearest
%	Neighbor Algorithm", IEEE Transactions on Systems, Man, and Cybernetics,
%	Vol. 15, No. 4, pp. 580-585.  
%
%	For selfdemo, type "initfknn" with no arguments.
%
%	See also FKNN.

%	Roger Jang, 990805

if nargin==0, selfdemo; return; end

feature_n = size(sampledata,2)-1;
data_n = size(sampledata,1);
sample_input = sampledata(:, 1:feature_n);
sample_output = sampledata(:, feature_n+1);
class_n = max(sample_output);

% Euclidean distance matrix
distmat = vecdist(sample_input);
distmat(1:(data_n+1):data_n^2) = inf;		% Set diagonal elements to infs

% knnmat(i,j) = class of i-th nearest point of j-th input vector
% (The size of knnmat is k times data_n.)
[junk, index] = sort(distmat);
knnmat = reshape(sample_output(index(1:k,:)), k, data_n);

% class_count(i,j) = count of class-i points within j-th test input vector's
% neighborhood
%class_count = zeros(class_n, test_n);
%for i = 1:test_n,
%	[sorted_element, element_count] = countele(knnmat(:,i));
%	class_count(sorted_element, i) = element_count;
%end

% Compute the membership grades for each sample point
fuz_class = zeros(data_n, class_n);
for i = 1:data_n,
	for j = 1:class_n
		desired_class = sampledata(i, end);
		n = length(find(knnmat(:,i)==j));
		if j == desired_class,
			fuz_class(i,j) = n/k*0.49+0.51;
		else
			fuz_class(i,j) = n/k*0.49;
		end
	end
end

% ========== Self Demo ==========
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
line([0 1], [0 1], 'linestyle', ':');
line([0.5 1], [0.5 0], 'linestyle', ':');

legend('Sample data: Class 1', 'Sample data: Class 2',...
	'Sample data: Class 3', -1);

k = 3;
fuz_out = initfknn(sampledata, k);
index = find(sum(fuz_out.^0.5, 2)~=1); 
line(x(index), y(index), 'linestyle', 'none', 'marker', 'o', 'color', 'w');

title('Circled data points have fuzzy membership grades.');

%for i = index(:)',
%	text(x(i), y(i), mat2str(fuz_out(i, :), 2));
%end
