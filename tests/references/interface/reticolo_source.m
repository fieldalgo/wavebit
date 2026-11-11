% RETICOLO's field of a dipole under a patterned layer, for gen_goldens_reticolo_rule_paths.py.
%
% res1 to res3 have no source, so this goes through what they call: retcouche for each texture,
% rets for the source, which retchamp takes as a thin row. No apodization, so the source is the
% delta function. c says whether the patterned layer's tables are built: 1 is res1's
% champ = 1, 0 its champ = 0.
%
% Reads `infile`: wavelength, period, order, x, y, n_top, n_bot, n_host, block ([cx cy dx dy n 1]),
% position, component, patterned, host_above, row, host_below, c. Saves e and z to `outfile`.
function reticolo_source(infile, outfile)
warning('off', 'all');
s = load(infile);
retss(3);
N = s.order;
k0 = 2 * pi / s.wavelength;
init = retinit(s.period, [-N, N, -N, N], [0, 0]);
init{end}.granet = 0;
top = retcouche(init, retu(init, {s.n_top, k0}), 1, 1);
bot = retcouche(init, retu(init, {s.n_bot, k0}), 1, 1);
host = retcouche(init, retu(init, {s.n_host, k0}), 1, 1);
patterned = retcouche(init, retu(init, {s.n_top, s.block, k0}), s.c, 1);
source = rets(init, s.position, host, s.component, [], 0);
sh = rettronc(retb(init, top, 1e-6), [], [], 1);
sb = rettronc(retb(init, bot, -1e-6), [], [], -1);
npts = -s.npts;
tab = [s.patterned, 2, npts; s.host_above, 1, npts; s.row, 1, 3i; s.host_below, 1, npts];
[e, z] = retchamp(init, {host, patterned, source}, sh, sb, 1, {s.x, s.y}, tab, [], 1:6);
out = struct('e', e, 'z', z);
save('-v7', outfile, '-struct', 'out');
end
