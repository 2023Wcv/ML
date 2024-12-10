function [y] =Fitness(x,data,trn,vald,classifierFhd)
    SzW=0.01;    
    x=logical(x);
    
    if sum(x(1:end))==0 
        x(1:end)=1;
    end
    
    numFeatures=sum(x(1:end));
    [outClass]  = classifierFhd(data(trn,:), data(trn,end), data(vald,:), data(vald,end),x);
    
    cp=classperf(data(vald,end),outClass);
    cp.CorrectRate;
    
    y=(1-SzW)*(1-cp.CorrectRate)+SzW*numFeatures/length(x(1:end));

end