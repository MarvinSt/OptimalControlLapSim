function Track = load_track_data(step_length,total_distance)

%% GET LAPDATA
file    = 'spa_&_ks_porsche_919_hybrid_2016_&_Player_&_stint_11.mat';
folder  = '';
data    = load(fullfile(folder, file));


%%
accy = smooth(data.CG_Accel_Lateral.Value .* 9.81, 50).';
velx = smooth(data.Ground_Speed.Value / 3.6, 50).';

% Compute lap distance
DIST = cumtrapz(velx .* gradient(data.Ground_Speed.Time));
CURV = accy ./ velx.^2;

% figure
% plot(data.CG_Accel_Lateral.Time, data.CG_Accel_Lateral.Value)
% grid on
% 
% figure
% plot(data.Ground_Speed.Time, data.Ground_Speed.Value)
% grid on
% 
% figure
% plot(data.Ground_Speed.Time, data.CG_Accel_Lateral.Value .* 9.81 ./ (data.Ground_Speed.Value / 3.6).^2)
% grid on
% 
% figure
% plot(DIST, (data.CG_Accel_Lateral.Value) .* 9.81 ./ (data.Ground_Speed.Value / 3.6).^2)
% grid on

%% GG
figure
plot3(data.CG_Accel_Lateral.Value, data.CG_Accel_Longitudinal.Value, data.Ground_Speed.Value, '.');
grid on

total_distance  = min(total_distance, DIST(end));
new_dist        = linspace(0, total_distance, 1 + total_distance / step_length);
new_curv        = interp1(DIST, CURV, new_dist);

dS              = mean(gradient(new_dist));
psi             = cumtrapz(dS .* new_curv);

Track.x = cumtrapz(dS.*cos(psi));
Track.y = cumtrapz(dS.*sin(psi));
Track.psi = psi;
Track.curv = new_curv;
Track.S = new_dist;
Track.N = length(Track.x);

end