function [Leader_pos]=bIPCACO(N,MaxFEs,dim,data,trn,vald,classifierFhd)
ub=1;
lb=0;
it=1;
k=45;          
q=0.7;
ibslo=0.8; 
m=N;
AllFitness = inf*ones(k,1);
bestFitness=inf;
%% 
X=initialization(k,dim,ub,lb)>0;
for i=1:k
    AllFitness(i)=Fitness(X(i,:),data,trn,vald,classifierFhd);
end
[~, SortOrder]=sort(AllFitness);
X=X(SortOrder,:);
bestPositions=X(1,:);                   
w=1/(sqrt(2*pi)*q*k)*exp(-0.5*(((1:k)-1)/(q*k)).^2);            
p_g=w/sum(w);           
%%
 while it<=MaxFEs   
    preGBestX = bestPositions;
    s=zeros(k,dim);
    for l=1:k
        s(l,:)=X(l,:);
    end  
    Newpop=zeros(m,dim);
    Newpop_fitness=inf*ones(m,1);
    for t=1:m     
        for i=1:dim 
            for j=1:k
                if p_g(j)>=rand
                      l=j;
                      break;
                end
            end     
            D=0;
            for r=1:k         
                D=D+abs(s(l,i)-s(r,i));
            end
            sigma=ibslo*D/(k-1);           
            temp = s(l,i)+sigma*randn;
            Newpop(t,i) = trnasferFun(s(l,i),temp);  
        end
        %% Evaluation
        Newpop_fitness(t)=Fitness(Newpop(t,:),data,trn,vald,classifierFhd);
        if Newpop_fitness(t) < bestFitness
            bestPositions = Newpop(t,:);
            bestFitness = Newpop_fitness(t);
        end
    end
    All_X=[X;Newpop];           
    All_X_fitness=[AllFitness;Newpop_fitness];
    [All_X_fitnessed, SortOrder]=sort(All_X_fitness);
    All_X=All_X(SortOrder,:);
    X=All_X(1:k,:);
    AllFitness=All_X_fitnessed(1:k);
    bestPositions = X(1,:);
    bestFitness = All_X_fitnessed(1);
    %% PID-based search
    Kp = 1;
    Ki = 0.5;
    Kd = 1.2;
    if it == 1
        Ek = bestPositions - X;
        Ek_1 = Ek;
        Ek_2 = Ek;
    else
        Ek_2 = Ek_1;
        Ek_1 = Ek + bestPositions - preGBestX;
        Ek = bestPositions - X;
        preGBestX = bestPositions;
    end
    a = (log(MaxFEs-it+2)/log(MaxFEs))^2;
    out0 = (cos(1-it/MaxFEs) + a*rand(k,dim).*Levy(k,dim)).*Ek;
    pid = rand(k,1)*Kp.*(Ek-Ek_1) + rand(k,1)*Ki.*Ek + rand(k,1)*Kd.*(Ek - 2*Ek_1 + Ek_2);

    r = rand(k,1)*cos(it/MaxFEs);
    PID_X = X + r.*pid + (1-r).*out0;
    for i = 1:k
        for j = 1:dim
            temp = PID_X(i,j);
            PID_X(i,j) = trnasferFun(X(i,j),temp);  
        end
        F_PID_i = Fitness(PID_X(i,:),data,trn,vald,classifierFhd);
        if F_PID_i < AllFitness(i)
            X(i,:) = PID_X(i,:);
            AllFitness(i) = F_PID_i;
            if AllFitness(i) < bestFitness
                bestFitness = AllFitness(i);
                bestPositions = X(i,:);
            end
        end
    end
    Leader_pos=bestPositions;
    it=it+1;
 end
end

%%  Levy search strategy
function o = Levy(n,d)
beta=1.5;
sigma=(gamma(1+beta).*sin(pi*beta/2)./(gamma((1+beta)/2).*beta.*2.^((beta-1)/2))).^(1/beta);
u=randn(n,d)*sigma;
v=randn(n,d);
step=u./abs(v).^(1/beta);
o=step;
end
