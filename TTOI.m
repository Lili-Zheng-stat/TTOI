function X_hat_arr=TTOI(Y_tensor,r_vec,iter,tol)
%Perform Tensor-Train Orthogonal Iteration (TTOI) algorithm for noisy observed order-d tensor Y_tensor, rank specified
%by r_vec. r_vec should be of dimension (d-1)*1. The maximum number of
%iterations is specified by iter, and the tolerance is specified by tol.
%The output X_hat_arr is a cell object consisting of the estimated tensors
%in all iterations.

dim_vec=size(Y_tensor);
d=length(dim_vec);
if length(r_vec)~=d-1
    fprintf('wrong number of ranks!\n');
    X_hat_arr=NaN;
else
    X_hat_arr=cell(iter,1);
    
    %initialize V_prod_arr, U_prod_arr, Y_arr (Y_1,...,Y_{d-1}).  
    V_prod_arr=cell(floor(iter/2)+1,1);
    U_prod_arr=cell(ceil(iter/2),1);
    Y_arr=cell(d-1,1);
    for i=1:(d-1)
        Y_arr{i}=reshape(Y_tensor,prod(dim_vec(1:i)),prod(dim_vec((i+1):d)));
    end
    
    %first V_prod all one's
    V_prod_arr{1}=cell(d-1,1);
    for i=1:(d-1)
        V_prod_arr{1}{i}=1;
    end
    
    %initialize iteration time (compared with iter) and change of estimators (compared with tol)
    n=0;chg=Inf;
    while (n<iter)&&(chg>tol)
        if mod(n,2)==0
            %perform sequential SVT from left to right, using
            %V_prod_arr{n/2+1} to compress the dimension
            U_prod_arr{n/2+1}=cell(d-1,1);
            %Y_tilde_arr saves calculation that can also be used in the next iteration
            Y_tilde_arr=cell(d-2,1);
            %estimate the first left singular space
            [U_temp,~,~]=svds(Y_arr{1}*V_prod_arr{n/2+1}{d-1},r_vec(1));
            U_prod_arr{n/2+1}{1}=U_temp;
            for k=2:(d-1)
                %Y_temp is the reshaped residual
                Y_temp=kron(eye(dim_vec(k)),U_prod_arr{n/2+1}{k-1})'*Y_arr{k};
                Y_tilde_arr{k-1}=reshape(Y_temp,r_vec(k-1),prod(dim_vec(k:d)));
                [U_temp,~,~]=svds(Y_temp*V_prod_arr{n/2+1}{d-k},r_vec(k));
                U_prod_arr{n/2+1}{k}=kron(eye(dim_vec(k)),U_prod_arr{n/2+1}{k-1})*U_temp;
            end
            X_hat_temp=U_prod_arr{n/2+1}{d-1}'*Y_arr{d-1};
            X_hat_arr{n+1}=reshape(U_prod_arr{n/2+1}{d-1}*X_hat_temp,dim_vec);
        else
            %perfrom sequential SVT from right to left, using
            %U_prod_arr{(n+1)/2} to compress the dimension
            V_prod_arr{(n+1)/2+1}=cell(d-1,1);
            %estimate the first right singular space
            [~,~,V_temp]=svds(U_prod_arr{(n+1)/2}{d-1}'*Y_arr{d-1},r_vec(d-1));
            V_prod_arr{(n+1)/2+1}{1}=V_temp;
            for k=2:(d-1)
                %perform SVT on the compressed residual
                [~,~,V_temp]=svds(Y_tilde_arr{d-k}*kron(V_prod_arr{(n+1)/2+1}{k-1},eye(dim_vec(d-k+1))),r_vec(d-k));
                V_prod_arr{(n+1)/2+1}{k}=kron(V_prod_arr{(n+1)/2+1}{k-1},eye(dim_vec(d-k+1)))*V_temp;
            end
            X_hat_temp=Y_arr{1}*V_prod_arr{(n+1)/2+1}{d-1};
            X_hat_arr{n+1}=reshape(X_hat_temp*V_prod_arr{(n+1)/2+1}{d-1}',dim_vec);
        end
        n=n+1;
        %update chg to be compared with tol
        if n>1
            chg=sum(X_hat_arr{n}(:).^2)-sum(X_hat_arr{n-1}(:).^2);
        end
    end
    %the total number of iteration is n
    X_hat_arr=X_hat_arr(1:n);    
end