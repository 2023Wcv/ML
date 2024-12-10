
function [fobj] = Get_FKNN()
   fobj = @fknnForFS; 
end

function predict_label = fknnForFS(train_fea, train_label, test_fea, ~,x)
    sub_fea_index = (x == 1);
    sub_train_fea = train_fea(:, sub_fea_index);
    sub_test_fea = test_fea(:, sub_fea_index);
    sampledata = [sub_train_fea,train_label];
    k = 1;
    m = 2;
    fuz_sample_out = initfknn(sampledata,k);
    [predict_label,~] = fknn(sub_train_fea,fuz_sample_out,sub_test_fea,k,m);
end

