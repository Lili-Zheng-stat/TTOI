function p_tensor=proj_simplex(X_tensor)
%Project fibers in the last mode of X_tensor onto the probability simplex.
%The projection algorithm is credited to the paper 
%''Efficient Projections onto the l1-Ball for Learning in High Dimensions''
%by John Duchi, Shai Shalev-Shwartz, Yoram Singer, Tushar Chandara

dim_vec=size(X_tensor);
d=length(dim_vec);
X_mat=reshape(X_tensor,[prod(dim_vec(1:(d-1))) dim_vec(d)]);
mu=sort(X_mat,2,'descend');
t=(cumsum(mu,2)-1)./repmat(1:dim_vec(d),[prod(dim_vec(1:(d-1))) 1]);
ind=sum(mu-t>0,2);
theta=zeros(prod(dim_vec(1:(d-1))),1);
for i=1:prod(dim_vec(1:(d-1)))
    theta(i)=t(i,ind(i));
end
p_tensor=max(X_tensor-reshape(repmat(theta,1,dim_vec(d)),dim_vec),0);
end
