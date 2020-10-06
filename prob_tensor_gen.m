function P_tensor=prob_tensor_gen(signal_sigma,d,dim_vec,r_vec)
%create a probability tensor with tt_decomposition structure: all entries are non-negative and the last mode sums up to 1 
%dimension vector: dim_vec of size d, ranks: r_vec of size d-1.
%signal_sigma controls the noise level for generating TT-factors G.

%First we generate each element of G as i.i.d N(0,signal_sigma^2), then we
%take their absolute values, and normalize it such that the last mode of
%each G_i sum up to 1.
G_arr=cell(d,1);
temp=normrnd(0,signal_sigma,dim_vec(1),r_vec(1));
G_arr{1}=abs(temp)./repmat(sum(abs(temp),2),1,r_vec(1));
temp=normrnd(0,signal_sigma,r_vec(d-1),dim_vec(d));
G_arr{d}=abs(temp)./repmat(sum(abs(temp),2),1,dim_vec(d));
for i=2:(d-1)
    temp=normrnd(0,signal_sigma,r_vec(i-1),dim_vec(i),r_vec(i));
    G_arr{i}=abs(temp)./repmat(sum(abs(temp),3),1,1,r_vec(i));
end
P_tensor=G_arr{1};
for k=1:(d-2)
    P_expand=repmat(reshape(P_tensor,[dim_vec(1:k),r_vec(k),1,1]),...
        [ones(1,k+1),dim_vec(k+1),r_vec(k+1)]);
    G_expand=repmat(reshape(G_arr{k+1},[ones(1,k),r_vec(k),dim_vec(k+1),r_vec(k+1)]),...
        [dim_vec(1:k),1,1,1]);
    P_tensor=reshape(sum(P_expand.*G_expand,k+1),[dim_vec(1:(k+1)),r_vec(k+1)]);
end
P_expand=repmat(reshape(P_tensor,[dim_vec(1:(d-1)),r_vec(d-1),1]),...
    [ones(1,d),dim_vec(d)]);
G_expand=repmat(reshape(G_arr{d},[ones(1,d-1),r_vec(d-1),dim_vec(d)]),...
    [dim_vec(1:(d-1)),1,1]);
P_tensor=reshape(sum(P_expand.*G_expand,d),dim_vec);
