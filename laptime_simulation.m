close all;clear;clc

addpath('casadi')

%% Vehicle Parameters:
m = 1500;    % vehicle mass (kg)
I_z = 1200;  % yaw inertia (kgm?)
lf = 0.8;   % distance COG to front axle (m)
lr = 0.8;   % distance COG to rear axle (m)
% Tire Parameters
A =15000; B =1.5; C =25; D =1; E =20;
% maximum Power
P_max = 6 * 80000;      % (W)

g = 9.81;


%% Track parameters
step_length     = 2;
total_distance  = 2000;
Track           = load_track_data(step_length,total_distance);
track_width     = 3;
step_length     = mean(gradient(Track.S));

%% Vehicle dynamics

% Define states
nx          = 5;
% nx          = length(states);

xnom        = [ 6 2 150 100 2 ].';
xnom        = [ 1 1 1 1 1 ].';

x_init      = [ 0 0 10 0 0 ].' ./ xnom;
lbx         = [ -track_width -60/180*pi 000 -50 -1  ].' ./ xnom;
ubx         = [ +track_width +60/180*pi 150 +50 +1  ].' ./ xnom;

xsym     	= casadi.SX.sym('states', nx);
x           = xsym .* xnom;

% Define controls
nu          = 3;

unom        = [ 1 1 1 ].';

u_init      = [ 0 0 0 ].' ./ unom;
lbu         = [ -30/180*pi -0.5 -0.5 ].' ./ unom;
ubu         = [ +30/180*pi +0.5 +0.5 ].' ./ unom;

usym      	= casadi.SX.sym('controls', nu);
upsym      	= casadi.SX.sym('prev_controls', nu);
u           = usym .* unom;
du          = (upsym - usym) .* unom;

% Get states
n       = x(1);   	% orthogonal path deviation (m)
xi      = x(2);    	% heading angle deviation (rad)
vx      = x(3);  	% vehicle fixed x-velocity (m/s)
vy      = x(4);    	% vehicle fixed y-velocity (m/s)
dpsi    = x(5);   	% vehicle yaw rate (rad/s)

% Get control inputs
delta   = u(1);  	% steering angle (rad)
sxf     = u(2);     % front long. slip (-)
sxr     = u(3);     % rear long. slip (-)

Qu      = diag([1e-3,1e-2,1e-2]);
Qdu     = diag([1e-1,1e-2,1e-2]);


% Curvature variable
Crv     = casadi.SX.sym('Crv');

% front/rear slip
syf     = -(cos(delta) * (vy + lf*dpsi) - sin(delta)*vx)/vx;
syr     = -(vy - lr*dpsi)/vx;

sf      = sqrt(sxf^2 + syf^2 + 1e-12);
sr      = sqrt(sxr^2 + syr^2 + 1e-12);

% Tire forces
% pure slip
Fxpf    = A*sin(B*atan(C*sf));
Fypf    = A*sin(B*atan(C*sf));
Fxpr    = A*sin(B*atan(C*sr));
Fypr    = A*sin(B*atan(C*sr));
% combined slip
Fxf     = Fxpf * sxf / sf;
Fyf     = Fypf * syf / sf;
Fxr     = Fxpr * sxr / sr;
Fyr     = Fypr * syr / sr;

Fdrg    = 0.65 * 1.22 * 0.5 * vx * abs(vx);

% Dynamics Scaling factor
Sf      = (1-n*Crv)/(vx*cos(xi)-vy*sin(xi));

% Spatial dynamics
dx      = casadi.SX.sym('rhs',nx);
dx(1)   = Sf * (vx*sin(xi) + vy*cos(xi));
dx(2)   = Sf * dpsi - Crv;
dx(3)   = Sf * (+dpsi*vy + 1/m * (Fxf*cos(delta) - Fyf*sin(delta) + Fxr - Fdrg));
dx(4)   = Sf * (-dpsi*vx + 1/m * (Fyf*cos(delta) + Fxf*sin(delta) + Fyr));
dx(5)   = Sf * 1/I_z*(lf*(Fyf*cos(delta) + Fxf*sin(delta)) - lr*Fyr);
f_rhs   = casadi.Function('f_rhs',{xsym,usym,Crv},{dx});

%% CONSTRAINTS
% approximation of power output
Power   = Fxf*(vx*cos(delta)+vy*sin(delta))*(1+sxf) + Fxr*vx*(1+sxr);
f_Power = casadi.Function('f_Power',{xsym,usym},{Power});
j_Cost  = Sf * Sf + u.' * Qu * u + du.' * Qdu * du; % + beta*beta*1e-4*0
f_Cost  = casadi.Function('f_Cost',{xsym,usym,upsym,Crv},{j_Cost});

%% INTEGRATOR
% Runge-Kutta 4 integration
M   = 1; % RK4 steps per interval
ds  = step_length/M;
X   = xsym;
% for i_ = 1:M
%     k1 = f_rhs(X,controls,Crv);
%     k2 = f_rhs(X+ds/2*k1,controls,Crv);
%     k3 = f_rhs(X+ds/2*k2,controls,Crv);
%     k4 = f_rhs(X+ds*k3,controls,Crv);
%     X = X + ds/6*(k1+2*k2+2*k3+k4);
% end
for i_  = 1 : M
    k1  = f_rhs(X,usym,Crv);
    k2  = f_rhs(X+ds*k1,usym,Crv);
    X   = X + ds/2*(k1+k2);
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