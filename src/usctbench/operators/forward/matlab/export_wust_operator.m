function export_wust_operator(folder, functions_path)
% Export upstream Helmholtz physics, not a Python reimplementation of its PDE.
% Freeze the optimized stencil range; otherwise differentiating H also needs
% the nonsmooth derivative of its extrema-dependent optimization coefficients.
addpath(functions_path);
in = load(fullfile(folder,'request.mat'));
[ny,nx] = size(in.speed_yx);
for index = 1:numel(in.frequencies)
    frequency = in.frequencies(index);
    solver = HelmholtzSolver(in.x, in.y, in.speed_yx, zeros(ny,nx), ...
        frequency, -1, in.pml_strength, in.pml_m, in.stencil_bounds);
    H = solver.HelmholtzEqn;
    PML = solver.PML;
    [~,d,e] = stencilOptParams(in.stencil_bounds(1),in.stencil_bounds(end), ...
        frequency, mean(diff(in.x)),mean(diff(in.y))/mean(diff(in.x)));
    rows = []; cols = []; values = [];
    for x = 2:nx-1
        for y = 2:ny-1
            for ox = -1:1
                for oy = -1:1
                    if ox == 0 && oy == 0
                        weight = 1-d-e;
                    elseif ox == 0 || oy == 0
                        weight = d/4;
                    else
                        weight = e/4;
                    end
                    rows(end+1) = sub2ind([ny nx],y,x); %#ok<AGROW>
                    cols(end+1) = sub2ind([ny nx],y+oy,x+ox); %#ok<AGROW>
                    values(end+1) = weight; %#ok<AGROW>
                end
            end
        end
    end
    B = sparse(rows,cols,values,ny*nx,ny*nx);
    save(fullfile(folder,sprintf('operator_%d.mat',index)),'H','B','PML','-v7');
end
end
