% RETICOLO's in-cell field with a permeability in a texture, for gen_goldens_reticolo_magnetic_field.py.
%
% Reads `infile` (a .mat the generator writes) and saves `outfile`, as reticolo_res3_2d.m does:
%
%   path                               '2d', 'te', 'tm' or 'conical'
%   wavelength, period, nn, k_parallel, delta0 (degrees), champ, npts, x, y
%   n_top, n_bot, n_layers, thickness  top to bottom, RETICOLO's profile order
%   bg_k, blocks_k                     2D: background index, rows [cx cy dx dy n 1]
%   edges_k, n_k                       1D: right edges and indices, from the cell's corner
%   n_ci, ci_k                         parm.res1.change_index, [n nx ny nz mx my mz]
%   n_einc, einc_k                     2D and conical incidences, [a_p, a_s]
%   sens                               optional: parm.res3.sens, -1 lights the stack from below
%
% Saves e_k (res3's raw cell for einc_k), z and o.
function reticolo_res3_magnetic(infile, outfile)
warning('off', 'all');
s = load(infile);
path = char(s.path);
textures = cell(1, s.n_layers + 2);
textures{1} = s.n_top;
textures{end} = s.n_bot;
for k = 1:s.n_layers
    if strcmp(path, '2d')
        tex = {s.(sprintf('bg_%d', k))};
        blocks = s.(sprintf('blocks_%d', k));
        for j = 1:size(blocks, 1)
            tex{end + 1} = blocks(j, :);
        end
    else
        tex = {s.(sprintf('edges_%d', k)), s.(sprintf('n_%d', k))};
    end
    textures{k + 1} = tex;
end

if strcmp(path, 'te')
    parm = res0(1);
elseif strcmp(path, 'tm')
    parm = res0(-1);
else
    parm = res0;
end
parm.not_io = 1;
parm.res1.champ = s.champ;
parm.res3.npts = s.npts * ones(1, s.n_layers + 2);
if isfield(s, 'sens')
    parm.res3.sens = s.sens;
end
parm.res1.change_index = cell(1, s.n_ci);
for k = 1:s.n_ci
    parm.res1.change_index{k} = s.(sprintf('ci_%d', k));
end
profile = {[0, s.thickness, 0], 1:(s.n_layers + 2)};
out = struct();
if strcmp(path, 'te') || strcmp(path, 'tm')
    aa = res1(s.wavelength, s.period, textures, s.nn, s.k_parallel, parm);
    [e, z, o] = res3(s.x, aa, profile, 1, parm);
    out.e_1 = e;
else
    aa = res1(s.wavelength, s.period, textures, s.nn, s.k_parallel, s.delta0, parm);
    for k = 1:s.n_einc
        [e, z, o] = res3(s.x, s.y, aa, profile, s.(sprintf('einc_%d', k)), parm);
        out.(sprintf('e_%d', k)) = e;
    end
end
out.z = z;
out.o = o;
save('-v7', outfile, '-struct', 'out');
end
