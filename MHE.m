% consider the process noise in MHE equality and inequality constraints 
% for two-dimensional system
% process noise: U(-1,1)

close all
clear all
clc 

%--------------------------------------------------------------------------
% MHE FOR LINEAR SYSTEM WITH QUADRATIC COST FUNCTION
% LINEAR SYSTEM IS : x(k+1) = A_s*x(k)+B_s*u(k)+w(k)
%                      y(k) = C_s*x(k)+v(k)
%--------------------------------------------------------------------------

global A_s B_s C_s
A_s = [0.99 0.2;
       -0.1 0.3];  
B_s = [0.00457;
       -0.00457];
C_s = [1 -3];
A_s_1 = A_s(1,:);
A_s_2 = A_s(2,:);

global N 
N = 16;           % horizon length of prediction
str = '16';
M = 100;         % number of time instance

x_1_D1 = zeros(M+1,1);
x_2_D1 = zeros(M+1,1);
x_1_D1(1) = 1;     % initial value of state
x_2_D1(1) = 19;

x_1_D2 = zeros(M+1,1);
x_2_D2 = zeros(M+1,1);
x_1_D2(1) = 1;     % initial value of state
x_2_D2(1) = 19;

x_1_D3 = zeros(M+1,1);
x_2_D3 = zeros(M+1,1);
x_1_D3(1) = 1;     % initial value of state
x_2_D3(1) = 19;

v = rand;
global y_D1 y_D2 y_D3
y_D1 = zeros(M+1,1);
y_D1(1) = C_s*[x_1_D1(1); x_2_D1(1)]+v;    % initial value of output
y_D2 = zeros(M+1,1);
y_D2(1) = C_s*[x_1_D2(1); x_2_D2(1)]+v;
y_D3 = zeros(M+1,1);
y_D3(1) = C_s*[x_1_D3(1); x_2_D3(1)]+v;

u = zeros(N+1,1);
x_hat_1_D1 = zeros(M,1);
x_hat_2_D1 = zeros(M,1);
x_hat_1_D2 = zeros(M,1);
x_hat_2_D2 = zeros(M,1);
x_hat_1_D3 = zeros(M,1);
x_hat_2_D3 = zeros(M,1);

b_old_D1 = [];             % initialize the constraints
b_old_D2 = [];
b_old_D3 = [];


% design the objective function for MHE
%--------------------------------------------------------------------------
D1 = zeros(2,2);
for i = 1:1:N
    D1 = D1+A_s^(i-1);    
end
D1 = diag([1 1]*D1);              % gradient matrix
D2 = eye(2);  
% sigma_norm = 1;                        % variance of process noise w
% D3 = sigma_norm*[1 1;1 1];
a = 0; 
b = 1;
sigma_uniform = (b-a)^2/12;          % variance of model uncertainty w
D3 = sigma_uniform*[1 1;1 1];
%--------------------------------------------------------------------------

u_D1  = zeros(M-1,1);
u_D2 = zeros(M-1,1);
u_D3 = zeros(M-1,1);

exitflag_xD1 = zeros(M,1);
exitflag_xD2 = zeros(M,1);
exitflag_xD3 = zeros(M,1);

exitflag_u1 = zeros(M,1);
exitflag_u2 = zeros(M,1);
exitflag_u3 = zeros(M,1);

for k = 1:M    
    Text = ['Step ',num2str(k),' of ',num2str(M),' MHE-EMPC Processing'];
    disp(Text)
    w = 2*rand-1;                                          % w~U(-1,1)
    % uniform distribution random noise
    % w = normrnd(0,sigma_norm);             % normal distribution random noise
    v = b*rand;
    
    v_h = b;
    v_l = a;
    w_h = 1;                                                  % upper bound of process noise 
    w_l = -1;                                                  % lower bound of process noise
    
    
    %----------------------------------------------------------
    % MHE
    % equality constrainsts 
    % x_hat(k+1) = A_s*x_hat(k)+B_s*u_k;

    if (k==1)             % no equality constraints
        Aeq = [];
        beq_old_D1 = [];
        beq_old_D2 = [];
        beq_old_D3 = [];
    else
        d = ones(k-1,1);
        Aeq = eye(2*k)-kron(diag(d,-1),A_s);
        Aeq = Aeq([3:2*k],:);
        Aeq = horzcat(Aeq,kron(-1,eye(2*(k-1))));                         % constraints on process noise
        beq_old_D1 = vertcat(beq_old_D1,B_s*u_D1(k-1));
        beq_old_D2 = vertcat(beq_old_D2,B_s*u_D2(k-1));
        beq_old_D3 = vertcat(beq_old_D3,B_s*u_D3(k-1));
    end
     
    %----------------------------------------------------------
    % inequality constrains
    % -100 <= x_hat <= 100
    % v_l+y(k) <= C_s*x_hat <= v_h+y(k)   
    % and constraints on initial estimation
    if (k==1)
         a_1 = [1 0;
                      -1 0;
                      0 1;
                      0 -1;
                      1 -3;
                      -1 3];
        A_1 = kron(eye(k),a_1);
        a_2 = [1 0;                       % constraints on initial estimation
                   -1 0;
                   0 1;
                   0 -1];
        A_2 =  kron([ones(1,1),zeros(1,k-1)],a_2);      
        A = vertcat(A_1,A_2);           
    else
         a_1 =  [1 0;
                      -1 0;
                      0 1;
                      0 -1;
                      1 -3;
                      -1 3];
        A_1 = kron(eye(k),a_1);
        A_1 = horzcat(A_1,zeros(size(A_1,1),2*(k-1)));

        a_2 = [1 0;                                       % constraints on initial estimation
                   -1 0;
                   0 1;
                   0 -1];
        A_2 = kron([1,zeros(1,2*k-2)],a_2);

        a_3 = a_2;                                      % constraints on process noise
        A_3 = kron([zeros(1,2*k-2),1],a_3);

            A = vertcat(A_1,A_2,A_3);
    end
      
    
    if (k==1)
        b_1_D1 = [100;
                           100;
                           100;
                           100;
                           y_D1(k)-v_l;
                           -y_D1(k)+v_h];
        b_3 = [1.05;                    % constraints on initial estimation
                   -0.95;
                   19.05;
                   -18.95];
        b_old_D1 = vertcat(b_1_D1,b_3);
        
        
        b_1_D2 = [100;
                           100;
                           100;
                           100;
                           y_D2(k)-v_l;
                           -y_D2(k)+v_h];
        b_3 = [1.05;                    % constraints on initial estimation
                   -0.95;
                   19.05;
                   -18.95];
        b_old_D2 = vertcat(b_1_D2,b_3);
        
        b_1_D3 = [100;
                           100;
                           100;
                           100;
                           y_D3(k)-v_l;
                           -y_D3(k)+v_h];
                       
        b_3 = [1.05;                    % constraints on initial estimation
                   -0.95;
                   19.05;
                   -18.95];
        b_old_D3 = vertcat(b_1_D3,b_3);
        
        
    else
        b_1_D1 = [100;
                       100;
                       100;
                       100;
                       y_D1(k)-v_l;
                       -y_D1(k)+v_h];
        b_old_D1 = vertcat(b_old_D1, b_1_D1);       
        b_2_D1 = [1.05;                    % constraints on initial estimation
                       -0.95;
                       19.05;
                       -18.95];           
        b_old_D1 = vertcat(b_old_D1,b_2_D1);        
        b_3_D1 = [ w_h;                  % constraints on process noise
                          -w_l;
                          w_h;
                          -w_l];
        b_old_D1 = vertcat(b_old_D1, b_3_D1);
        
        b_1_D2 = [100;
                       100;
                       100;
                       100;
                       y_D2(k)-v_l;
                       -y_D2(k)+v_h];
        b_old_D2 = vertcat(b_old_D2, b_1_D2);         
        b_2_D2 = [1.05;                    % constraints on initial estimation
                       -0.95;
                       19.05;
                       -18.95];
        b_old_D2 = vertcat(b_old_D2,b_2_D2);       
        b_3_D2 = [ w_h;                  % constraints on process noise
                          -w_l;
                          w_h;
                          -w_l];
        b_old_D2 = vertcat(b_old_D2, b_3_D2);

        b_1_D3 = [100;
                       100;
                       100;
                       100;
                       y_D3(k)-v_l;
                       -y_D3(k)+v_h];
        b_old_D3 = vertcat(b_old_D3, b_1_D3); 
        b_2_D3 = [1.05;                    % constraints on initial estimation
                       -0.95;
                       19.05;
                       -18.95];
        b_old_D3 = vertcat(b_old_D3,b_2_D3);      
        b_3_D3 = [ w_h;                  % constraints on process noise
                          -w_l;
                          w_h;
                          -w_l];
        b_old_D3 = vertcat(b_old_D3, b_3_D3);
        
    end
                      
       
%     x0 = 0.001*ones(2*k,1);                   % starting point
    if (k==1)
       x0_D1 = [x_1_D1(k);x_2_D1(k)];
       x0_D2 = [x_1_D2(k);x_2_D2(k)]; 
       x0_D3 = [x_1_D3(k);x_2_D3(k)];      
    else
        if (k==2)
            x0_D1 = [x_hat_D1(1:2*(k-1));1;1;1;1];
            x0_D2 = [x_hat_D2(1:2*(k-1));1;1;1;1];
            x0_D3 = [x_hat_D3(1:2*(k-1));1;1;1;1];        
        else 
           x0_D1 = [x_hat_D1(1:2*(k-1));1;1;x_hat_D1(2*k-1:end);1;1];
           x0_D2 = [x_hat_D2(1:2*(k-1));1;1;x_hat_D1(2*k-1:end);1;1];
           x0_D3 = [x_hat_D3(1:2*(k-1));1;1;x_hat_D1(2*k-1:end);1;1];                      
        end
   
    end

    lb = [];
    ub = [];
    nonlcon = [];

%     options=optimset('disp','off','LargeScale','off','TolFun',.001,'MaxIter',5000,'MaxFunEvals',60000);
    options=optimset('disp','off','LargeScale','off','TolX',1e-10,'MaxIter',5000,'MaxFunEvals',60000);
    
    [x_hat_D1,fval,exitflag_x1,output]  = fmincon(@(x_hat)obj_mhe(x_hat,D1,k),x0_D1,A,b_old_D1,Aeq,beq_old_D1,lb,ub,nonlcon,options);
        
    [x_hat_D2,fval,exitflag_x2,output] = fmincon(@(x_hat)obj_mhe(x_hat,D2,k),x0_D2,A,b_old_D2,Aeq,beq_old_D2,lb,ub,nonlcon,options);
    
    [x_hat_D3,fval,exitflag_x3,output] = fmincon(@(x_hat)obj_mhe(x_hat,D3,k),x0_D3,A,b_old_D3,Aeq,beq_old_D3,lb,ub,nonlcon,options);
    
    b_old_D1 = b_old_D1(1:6*k,:);
    b_old_D2 = b_old_D2(1:6*k,:);
    b_old_D3 = b_old_D3(1:6*k,:);
     
    x_hat_1_D1(k) = x_hat_D1(2*k-1);
    x_hat_2_D1(k) = x_hat_D1(2*k);
         
    x_hat_1_D2(k) = x_hat_D2(2*k-1);
    x_hat_2_D2(k) = x_hat_D2(2*k);
   
    x_hat_1_D3(k) = x_hat_D3(2*k-1);
    x_hat_2_D3(k) = x_hat_D3(2*k);
    
    exitflag_xD1(k) = exitflag_x1;
    exitflag_xD2(k) = exitflag_x2;
    exitflag_xD3(k) = exitflag_x3; 
    
    %----------------------------------------------------------------------
    %----------------------------------------------------------------------
    % EMPC
    % eqality constrains
    % x_hat(k+1) = A_s*x_hat(k)+B_s*u(k);
%     Aeq = [];
%     beq = [];
    %------------------------------------
%     % use nonlinear constraints
%     nonlcon = @(x)statecon(x,k);
%     %------------------------------------
    % use linear constraints
    nonlcon = [];
    d = ones(N-1,1);
    Aeq_1 = eye(2*N)-kron(diag(d,-1),A_s);
    Aeq_1 = Aeq_1([3:2*N],:);
    Aeq_2 = kron(eye(N-1),B_s);
    Aeq = horzcat(Aeq_1,Aeq_2);
    beq = zeros(size(Aeq,1),1);

    
    %-----------------------------------------------------------------------------
    % inequality constrainst
    % -10 <= u <= 10
    % -100 <= x_hat <= 100     
    A_u = [];
    b_u = [];
    
    lb_D1 = [x_hat_1_D1(k)-0.05;x_hat_2_D1(k)-0.05;-100*ones(2*N-2,1);-10*ones(N-1,1)];   % N constraints on states and (N-1) constraints on u
    ub_D1 = [x_hat_1_D1(k)+0.05;x_hat_2_D1(k)+0.05;100*ones(2*N-2,1);10*ones(N-1,1)];
    lb_D2 = [x_hat_1_D2(k)-0.05;x_hat_2_D2(k)-0.05;-100*ones(2*N-2,1);-10*ones(N-1,1)]; 
    ub_D2 = [x_hat_1_D2(k)+0.05;x_hat_2_D2(k)+0.05;100*ones(2*N-2,1);10*ones(N-1,1)];
    lb_D3 = [x_hat_1_D3(k)-0.05;x_hat_2_D3(k)-0.05;-100*ones(2*N-2,1);-10*ones(N-1,1)];
    ub_D3 = [x_hat_1_D3(k)+0.05;x_hat_2_D3(k)+0.05;100*ones(2*N-2,1);10*ones(N-1,1)];

    if (k==1)
        x0_empc_D1 = [zeros(2*N,1);zeros(N-1,1)];
        x0_empc_D2 = [zeros(2*N,1);zeros(N-1,1)];
        x0_empc_D3 = [zeros(2*N,1);zeros(N-1,1)];
    else
        x0_empc_D1 = [u1(1:2*N);ones(N-1,1)];
        x0_empc_D2 = [u2(1:2*N);ones(N-1,1)];
        x0_empc_D3 = [u3(1:2*N);ones(N-1,1)];      
    end
      
    options=optimset('disp','off','LargeScale','off','TolFun',.001,'MaxIter',5000,'MaxFunEvals',60000);
    
   [u1,fval,exitflag_uD1,output] = fmincon(@(x)obj_empc(x),x0_empc_D1,A_u,b_u,Aeq,beq,lb_D1,ub_D1,nonlcon,options);
    u_D1(k) = u1(2*N+1);
    
    [u2,fval,exitflag_uD2,output] = fmincon(@(x)obj_empc(x),x0_empc_D2,A_u,b_u,Aeq,beq,lb_D2,ub_D2,nonlcon,options);
    u_D2(k) = u2(2*N+1);
    
    [u3,fval,exitflag_uD3,output] = fmincon(@(x)obj_empc(x),x0_empc_D3,A_u,b_u,Aeq,beq,lb_D3,ub_D3,nonlcon,options);
    u_D3(k) = u3(2*N+1);

    x_D1 = A_s*[x_hat_1_D1(k); x_hat_2_D1(k)]+B_s*u_D1(k)+w;          % updata real value
    x_1_D1(k+1) = x_D1(1);
    x_2_D1(k+1) = x_D1(2);
    
    x_D2 = A_s*[x_hat_1_D2(k); x_hat_2_D2(k)]+B_s*u_D2(k)+w;
    x_1_D2(k+1) = x_D2(1);
    x_2_D2(k+1) = x_D2(2);
    
    x_D3 = A_s*[x_hat_1_D3(k); x_hat_2_D3(k)]+B_s*u_D3(k)+w;
    x_1_D3(k+1) = x_D3(1);
    x_2_D3(k+1) = x_D3(2);

    y_D1(k+1) = C_s*[x_1_D1(k+1);x_2_D1(k+1)]+v;    
    y_D2(k+1) = C_s*[x_1_D2(k+1);x_2_D2(k+1)]+v; 
    y_D3(k+1) = C_s*[x_1_D3(k+1);x_2_D3(k+1)]+v; 
    
    exitflag_u1(k) = exitflag_uD1;
    exitflag_u2(k) = exitflag_uD2;
    exitflag_u3(k) = exitflag_uD3;
      
    cost_1(k) = 0;
    cost_1(k) = sum(x_1_D1(1:k+1))+sum(x_2_D1(1:k+1));
    cost_1(k) = cost_1(k)/k;          % average economic cost
    
    cost_2(k) = 0;
    cost_2(k) = sum(x_1_D2(1:k+1))+sum(x_2_D2(1:k+1));
    cost_2(k) = cost_2(k)/k;
    
    cost_3(k) = 0;
    cost_3(k) = sum(x_1_D3(1:k+1))+sum(x_2_D3(1:k+1));
    cost_3(k) = cost_3(k)/k;
    
end

% trajectory of state estimation
figure
stem(x_hat_1_D1,'*','blue');
hold on
stem(x_hat_1_D2,'*','black');
hold on
stem(x_hat_1_D3,'*','green');
hold on
stem(x_1_D1,'filled','red');
legend('estimation value','estimation value when D is eye(2)','estimation value when D var(w)','real value');
xlabel('iteration'); 
title(strcat('Estiamtion & real value of the first state variable when the horizon is N=',str));

% evaluate performance of three functions
figure
plot(cost_1,'blue');
hold on
plot(cost_2,'red');
hold on
plot(cost_3,'green');
legend('D = gradient','D = eye(2)','D = var(w)');
title('Comparison of economic costs');

% figure
% plot(u_D1,'blue');
% hold on
% plot(u_D2,'red');
% hold on
% plot(u_D3,'green');
% legend('D = gradient','D = eye(2)','D = var(w)');
% title('Control effort');

function f1 = obj_mhe(x_hat,D,k)
    global y_D1 y_D2 y_D3 C_s
    n_w = x_hat(2*k+1:end);
    n_w_p = n_w.^2;
    n_v = [];
    for c = 1:1:k-1
        n_v =[n_v; y_D1(c)-C_s*[x_hat(2*c-1);x_hat(2*c)]];
    end
    n_v_p = n_v.^2;
    f1 = [x_hat(2*k-1) x_hat(2*k)]*D*[x_hat(2*k-1);x_hat(2*k)]+sum(n_w_p)+sum(n_v_p);
end

function f2 = obj_empc(x)   
    global N
    x_hat = x(1:2*N);
    f2 = 0;
    for i = 1:1:N
       f2 = f2+[1 1]*[x_hat(2*i-1);x_hat(2*i)];
    end
end

% function [c,ceq] = statecon(x,k)
%     global A_s
%     global B_s
%     global N
%     global x_hat_1_D1
%     global x_hat_2_D1    
%     c = [];
%     ceq = [x_hat_1_D1(k);x_hat_2_D1(k)];
%     x_hat = x(1:2*N);
%     u = x(2*N+1:end);    
%     for j = 2:1:N-1
%         ceq = vertcat(ceq,x_hat(2*(j+1)-1:2*(j+1))-A_s*x_hat(2*j-1:2*j)-B_s*u(j));
%     end
% end




