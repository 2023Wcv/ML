function [posOut]=trnasferFun(pos,temp)
    s=abs((2/pi)*atan((pi/2)*temp));
    if rand<s 
       posOut=~pos; 
     else
       posOut=pos;
    end    
end
      