function kwave_arrival_calibration(input_path, output_path, kwave_path, device)
% Uniform, independently known media. No breast image or inverse algorithm.
addpath(kwave_path);
in = load(input_path);
N = double(in.N);
kgrid = kWaveGrid(N, in.dx, N, in.dx);
kgrid.setTime(double(in.Nt), in.dt);
sensor.mask = zeros(N, N);
ids = sub2ind([N N], in.elements_yx(:,2)+1, in.elements_yx(:,1)+1);
sensor.mask(ids) = 1;
[~, order] = sort(ids);
nrx = numel(ids);
ntx = numel(in.tx_indices);
speeds = double(in.speeds);
pressure = zeros(kgrid.Nt, ntx, nrx, numel(speeds), 'single');
for k = 1:numel(speeds)
    if isfield(in, 'bump_yx')
        medium.sound_speed = (1500 + (speeds(k)-1500) * in.bump_yx).';
    else
        medium.sound_speed = speeds(k);
    end
    medium.density = 1000;
    for s = 1:ntx
        source.p_mask = zeros(N, N);
        source.p_mask(ids(in.tx_indices(s)+1)) = 1;
        source.p = in.source_signal;
        source.p_mode = 'additive';
        raw = kspaceFirstOrder2DG(kgrid, medium, source, sensor, ...
            'PMLSize', double(in.pml), 'PMLInside', true, 'PlotSim', false, ...
            'DataCast', 'single', 'DeviceNum', double(device), ...
            'DataPath', fileparts(output_path), 'DeleteData', true);
        reordered = zeros(nrx, kgrid.Nt, 'single');
        reordered(order,:) = raw;
        pressure(:,s,:,k) = reordered.';
        fprintf('Uniform c=%g TX %d/%d complete\n', speeds(k), s, ntx);
    end
end
time = kgrid.t_array;
save(output_path, 'pressure', 'time', 'speeds', '-v7.3');
end
