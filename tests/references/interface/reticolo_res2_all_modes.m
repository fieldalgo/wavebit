function [top_refl_info, top_tran_info] = reticolo_res2_all_modes(_pol, theta, period, n_inc, nn, _textures, _profile, wavelength, tolerance);
% reticolo_res2 for a 1D grating, with RETICOLO's mode selection opened to a complex k_parallel.
%
% res2 reports only the modes it counts as propagating (parm.res2.tolh / tolb, 1e-6 by
% default), and at a complex k_parallel -- or in an absorbing incident medium -- none is: the
% order lists come back empty, with no error. The tolerances reach res2 only as its *third*
% argument; set on the parm handed to res1 they are ignored without a word, which is how the
% first attempt at this returned empty lists at a tolerance of 100.

warning('off', 'Octave:possible-matlab-short-circuit-operator');
warning('off', 'Invalid UTF-8 byte sequences have been replaced.');
warning('off', 'findstr is obsolete; use strfind instead');

% A complex theta gives the formal n*sin(theta), the same parameterization the solver keeps.
k_parallel = n_inc*sin(theta);

if _pol == 0  # TE
    pol = 1;
else  # TM
    pol = -1;
end

parm = res0(pol);
parm.not_io = 1;  % no write data on hard disk
parm.res2.tolh = tolerance;
parm.res2.tolb = tolerance;

textures = cell(1, size(_textures, 2));
for i = 1:length(_textures)
    textures(i) = _textures(i);
end

profile = cell(1, size(_profile, 1));
profile(1) = _profile(1, :);
profile(2) = _profile(2, :);

aa = res1(wavelength, period, textures, nn, k_parallel, parm);
res = res2(aa, profile, parm);

top_refl_info = res.inc_top_reflected;
top_tran_info = res.inc_top_transmitted;
end
