function X_tensor=tensor_gen(signal_sigma,d,dim_vec,r_vec)
%create a tensor with tt_decomposition structure, 
%dimension vector: dim_vec of size d, ranks: r_vec of size d-1.
%each element of G are i.i.d N(0,signal_sigma^2).

G_arr=cell(d,1);
G_arr{1}=normrnd(0,signal_sigma,dim_vec(1),r_vec(1));
G_arr{d}=normrnd(0,signal_sigma,r_vec(d-1),dim_vec(d));
for i=2:(d-1)
    G_arr{i}=normrnd(0,signal_sigma,r_vec(i-1),dim_vec(i),r_vec(i));
end
X_tensor=G_arr{1};
for k=1:(d-2)
    X_expand=repmat(reshape(X_tensor,[dim_vec(1:k),r_vec(k),1,1]),...
        [ones(1,k+1),dim_vec(k+1),r_vec(k+1)]);
    G_expand=repmat(reshape(G_arr{k+1},[ones(1,k),r_vec(k),dim_vec(k+1),r_vec(k+1)]),...
        [dim_vec(1:k),1,1,1]);
    X_tensor=reshape(sum(X_expand.*G_expand,k+1),[dim_vec(1:(k+1)),r_vec(k+1)]);
end
X_expand=repmat(reshape(X_tensor,[dim_vec(1:(d-1)),r_vec(d-1),1]),...
    [ones(1,d),dim_vec(d)]);
G_expand=repmat(reshape(G_arr{d},[ones(1,d-1),r_vec(d-1),dim_vec(d)]),...
    [dim_vec(1:(d-1)),1,1]);
X_tensor=reshape(sum(X_expand.*G_expand,d),dim_vec);
