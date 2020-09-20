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
