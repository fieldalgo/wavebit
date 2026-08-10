function out = reti_aniso(wl, period, thick, eps_re, eps_im, theta_deg)
% One uniform anisotropic slab between vacuum half-spaces, via RETICOLO.
%
% Follows the shipped exemple_V9_0D_anisotrope.m: a texture given as a cell holding a
% full 3x3 complex permittivity tensor, and nn = 0 so only the zeroth order exists.
% eps is passed as separate real and imaginary parts because oct2py does not carry
% complex matrices across reliably.
%
% Returns [R_TE, T_TE, R_TM, T_TM].

parm = res0;
parm.not_io = 1;

nn = 0;                       % no structure -> zeroth order only
nh = 1;                       % vacuum above
ro = nh * sind(theta_deg);
delta0 = 0;

epsilon = eps_re + 1i * eps_im;

textures = cell(1, 3);
textures{1} = nh;
textures{2} = { epsilon };
textures{3} = 1;              % vacuum below

aa = res1(wl, period, textures, nn, ro, delta0, parm);

profil = {[0, thick, 0], [1, 2, 3]};
ef = res2(aa, profil);

out = [sum(ef.TEinc_top_reflected.efficiency), ...
       sum(ef.TEinc_bottom_transmitted.efficiency), ...
       sum(ef.TMinc_top_reflected.efficiency), ...
       sum(ef.TMinc_bottom_transmitted.efficiency)];
end
