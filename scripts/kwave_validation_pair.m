function kwave_validation_pair(input_path, output_path, kwave_path, device)
% Independent 2-D lossless k-Wave pressure pair for a bounded validation case.
% Python input images are [y,x]. k-Wave medium/source arrays below are [x,y].
% No inverse solver, source calibration or GT-derived ToF is used here.
addpath(kwave_path);
in = load(input_path);
N = size(in.sim_speed_yx, 1);
kgrid = kWaveGrid(N, in.dx, N, in.dx);
medium.sound_speed = in.sim_speed_yx.';
medium.density = 1000;
kgrid.makeTime(medium.sound_speed, 0.2, in.end_time_s);
source_signal = toneBurst(1/kgrid.dt, in.source_frequency_hz, 3);
ids = sub2ind([N N], in.elements_yx(:, 2)+1, in.elements_yx(:, 1)+1);
[~, order] = sort(ids);
sensor.mask = zeros(N, N);
sensor.mask(ids) = 1;
n = numel(ids);
full_dataset = zeros(kgrid.Nt, n, n, 'single');
water_dataset = full_dataset;
for tx = 1:n
    source.p_mask = zeros(N, N);
    source.p_mask(ids(tx)) = 1;
    source.p = source_signal;
    source.p_mode = 'additive';
    for reference = 0:1
        if reference
            medium.sound_speed = 1500;
        else
            medium.sound_speed = in.sim_speed_yx.';
        end
        raw = kspaceFirstOrder2DG(kgrid, medium, source, sensor, ...
            'PMLSize', double(in.pml), 'PMLInside', true, 'PlotSim', false, ...
            'DataCast', 'single', 'DeviceNum', double(device), ...
            'DataPath', fileparts(output_path), 'DeleteData', true);
        reordered = zeros(n, kgrid.Nt, 'single');
        reordered(order,:) = raw;
        if reference
            water_dataset(:,:,tx) = reordered.';
        else
            full_dataset(:,:,tx) = reordered.';
        end
    end
    fprintf('Completed pressure/water TX %d/%d\n', tx, n);
end
time = kgrid.t_array;
transducerPositionsXY = in.positions_yx.';
transducerPositionsXY = transducerPositionsXY([2 1],:);
xi_orig = in.image_x;
yi_orig = in.image_y;
C = in.image_speed_yx.';
simulation_dx = in.dx;
simulation_shape = [N N];
simulation_cfl = max(in.sim_speed_yx(:))*kgrid.dt/in.dx;
simulation_ppw = min(in.sim_speed_yx(:))/(in.dx*in.maximum_frequency_hz);
pml_pixels = in.pml;
save(output_path, 'full_dataset', 'water_dataset', 'time', ...
    'transducerPositionsXY', 'C', 'xi_orig', 'yi_orig', 'source_signal', ...
    'simulation_dx', 'simulation_shape', 'simulation_cfl', 'simulation_ppw', ...
    'pml_pixels', '-v7.3');
end
