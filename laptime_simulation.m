close all;clear;clc

addpath('casadi')


%% Track parameters
step_length     = 2;
total_distance  = 2000;
Track           = load_track_data(step_length,total_distance);
track_width     = 3;
step_length     = mean(gradient(Track.S));

%% DYNAMICS
% Load model here
bicycle_model

%% INTEGRATOR
% Runge-Kutta 4 integration or trapezoidal
RK  = 4;

M   = 1; % Number of integrations steps per interval
ds  = step_length/M;
X   = xsym;
if RK == 4
    for i_ = 1:M
        k1 = f_rhs(X,controls,Crv);
        k2 = f_rhs(X+ds/2*k1,controls,Crv);
        k3 = f_rhs(X+ds/2*k2,controls,Crv);
        k4 = f_rhs(X+ds*k3,controls,Crv);
        X = X + ds/6*(k1+2*k2+2*k3+k4);
    end
else
    for i_  = 1 : M
        k1  = f_rhs(X,usym,Crv);
        k2  = f_rhs(X+ds*k1,usym,Crv);
        X   = X + ds/2*(k1+k2);
    end
end
fX      = casadi.Function('fX',{xsym,usym,Crv},{X});


%% build optimization problem
% constraints setup
args        = struct;
% Number of discretization intervals
N           = Track.N-1;    
% Decision variables (controls)
U           = casadi.MX.sym('U',nu,1,N); 
% Decision variables (states)
X           = casadi.MX.sym('X',nx,1,N+1);

% Integration constraints
F_eom       = fX.map(N, 'thread', 8);
Xn          = F_eom([ X{1:N}  ], [ U{:} ], [ Track.curv(1:N) ]);
g           = vec([X{1}-x_init [ X{2:N+1} ] - Xn(:,1:N) ] ./ xnom);
args.lbg    = zeros(nx * (N+1), 1);
args.ubg    = zeros(nx * (N+1), 1);

% Power and additional constraints
F_pow       = f_Power.map(N, 'thread', 8);
Pw          = F_pow([ X{1:N} ], [ U{:} ]);
g           = [ g; vec(Pw - P_max) / P_max];
args.lbg    = [ args.lbg; -inf*ones(1 * N, 1); ];
args.ubg    = [ args.ubg; zeros(1 * N, 1); ];

% Objective function (minimize squared laptime and regularize inputs)
F_cost      = f_Cost.map(N, 'thread', 8);
J_cost      = F_cost([ X{1:N} ], [ U{:} ], [ u_init U{1:N-1} ], [ Track.curv(1:N) ]);
obj         = sum(J_cost);

% make the decision variables one column vector
vars        = [vec([U{:}]); vec([X{:}])];
args.lbx    = [ repmat(lbu, N, 1); repmat(lbx, N + 1, 1) ];
args.ubx    = [ repmat(ubu, N, 1); repmat(ubx, N + 1, 1) ];
args.x0     = [ repmat(u_init, N, 1); repmat(x_init, N + 1, 1) ];
nlp         = struct('f', obj, 'x', vars, 'g', g);

% Solver settings
opts = struct;
% opts.jit = true;
% opts.expand = true;
% opts.print_time = 0;
% opts.ipopt.max_iter = 100;
% opts.ipopt.print_level =0;%0,3
% opts.ipopt.acceptable_tol =1e-2;
% opts.ipopt.acceptable_obj_change_tol = 1e-6;
% opts.ipopt.nlp_scaling_method = 'none';
opts.ipopt.derivative_test = 'none';
opts.ipopt.max_iter = 5000;
% opts.ipopt.hessian_approximation = 'limited-memory';
opt.ipopt.fixed_variable_treatment =  'relax_bounds';

solver = casadi.nlpsol('solver', 'ipopt', nlp, opts);




%% Solve OCP
sol = solver('x0', args.x0, 'lbx', args.lbx, 'ubx', args.ubx,'lbg', args.lbg, 'ubg', args.ubg);
x_sol = full(sol.x);

% extract result
res.U = x_sol(1:nu*N);
res.U = reshape(res.U,nu,N);
res.X = x_sol(nu*N+1:end);
res.X = reshape(res.X,nx,N+1);

% calculate laptime
res.dt = step_length * (1-res.X(1,:).*Track.curv)./(res.X(3,:).*cos(res.X(2,:))-res.X(4,:).*sin(res.X(2,:)));
res.time = [0,cumsum(res.dt(1:N))];
res.laptime = res.time(end);
%% plots

%% path
figure('Color','w');
hold on;
stem3(Track.x-sin(Track.psi + res.X(2,:)).*res.X(1,:),Track.y+cos(Track.psi + res.X(2,:)).*res.X(1,:),sqrt(res.X(3,:).^2+res.X(3,:).^2));
plot(Track.x-sin(Track.psi)*track_width,Track.y+cos(Track.psi)*track_width,'-k','LineWidth',1);
plot(Track.x+sin(Track.psi)*track_width,Track.y-cos(Track.psi)*track_width,'-k','LineWidth',1);
plot(Track.x,Track.y,'--k','LineWidth',0.5);
grid on;
xlabel('x (m)','FontSize',14),ylabel('y (m)','FontSize',14);
legend({'vehicle path','reference path'},'FontSize',14);
daspect([1,1,1])

%% velocities
figure('Color','w');
plot(Track.S,res.X(3,:),'LineWidth',1);hold on;
plot(Track.S,res.X(4,:),'LineWidth',1);grid on;
xlabel('distance (m)','FontSize',14),ylabel('velocity (m/s)','FontSize',14);
legend({'v_x','v_y'},'FontSize',14);

%% normal path deviation
figure('Color','w');
plot(Track.S,res.X(1,:),'LineWidth',1);grid on;
xlabel('distance (m)','FontSize',14),ylabel('n (m)','FontSize',14);

%% steering angle
figure('Color','w');
plot(Track.S(1:length(res.U(1,:))),res.U(1,:)*180/pi,'LineWidth',1);grid on;
xlabel('distance(m)','FontSize',14),ylabel('steering angle (deg)','FontSize',14);

%% longitudinal slip
figure('Color','w');
plot(Track.S(1:length(res.U(2,:))),res.U(2,:),'LineWidth',1);hold on;
plot(Track.S(1:length(res.U(3,:))),res.U(3,:),'LineWidth',1);grid on;
xlabel('distance (m)','FontSize',14),ylabel('longitudinal slip','FontSize',14);
legend({'front slip','rear slip'},'FontSize',14);

%% yaw angle
% figure('Color','w');
% plot(Track.S,res.X(8,:)*180/pi,'LineWidth',1);hold on;
% plot(Track.S,Track.psi*180/pi,'LineWidth',1);grid on;
% xlabel('distance(m)','FontSize',14),ylabel('yaw angle (deg)','FontSize',14);
% legend({'vehicle','reference'},'FontSize',14);

%% power output
% approximation of power output
for k = 1:N
    res.Power(k) =  full(f_Power(res.X(:,k),res.U(:,k)));
end

figure('Color','w');
plot(Track.S(1:length(res.Power)),res.Power/1000,'LineWidth',1); grid on
xlabel('distance (m)','FontSize',14); ylabel('Power (kW)','FontSize',14);