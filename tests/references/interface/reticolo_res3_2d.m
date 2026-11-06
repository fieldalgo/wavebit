% RETICOLO's in-cell field of a crossed (2D) grating, for gen_goldens_reticolo_2d_field.py.
%
% Reads `infile` (a .mat the generator writes) and saves `outfile`. Driven through Octave as a
% subprocess rather than oct2py, so the cell arrays a 2D texture needs are built here, in
% RETICOLO's own language, from plain arrays:
%
%   wavelength, period (1x2), nn (1x2), k_parallel, delta0 (degrees), champ
%   n_top, n_bot                       ambient refractive indices
%   n_layers, thickness (1 x n_layers) top to bottom, RETICOLO's profile order
%   bg_k, blocks_k  (k = 1..n_layers)  background index, and rows [cx cy dx dy n 1]
%   x, y, npts                         sample lines and points per profile entry
%   einc_k (k = 1..n_einc), n_einc     incidences, [a_p, a_s] in res3's convention
%
% Saves e_k (res3's raw (z, x, y, 6) cell for einc_k), z and o.
function reticolo_res3_2d(infile, outfile)
warning('off', 'all');
s = load(infile);
textures = cell(1, s.n_layers + 2);
textures{1} = s.n_top;
for k = 1:s.n_layers
    tex = {s.(sprintf('bg_%d', k))};
    blocks = s.(sprintf('blocks_%d', k));
    for j = 1:size(blocks, 1)
        tex{end + 1} = blocks(j, :);
    end
    textures{k + 1} = tex;
end
textures{end} = s.n_bot;

parm = res0;
parm.not_io = 1;
parm.res1.champ = s.champ;
parm.res3.npts = s.npts * ones(1, s.n_layers + 2);
aa = res1(s.wavelength, s.period, textures, s.nn, s.k_parallel, s.delta0, parm);
profile = {[0, s.thickness, 0], 1:(s.n_layers + 2)};
out = struct();
for k = 1:s.n_einc
    [e, z, o] = res3(s.x, s.y, aa, profile, s.(sprintf('einc_%d', k)), parm);
    out.(sprintf('e_%d', k)) = e;
end
out.z = z;
out.o = o;
save('-v7', outfile, '-struct', 'out');
end
