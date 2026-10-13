"""The reported main-state payload equals the arrays actually initialized.

This is an allocation oracle, not a repetition of the report's pole-count
formula. No field step or time-domain solve is needed. Sources are absent
because source histories/auxiliary columns are outside this report field;
observer and energy-diagnostic storage is excluded explicitly as well.
"""
import warnings

import pytest

import wavebit.fdtd._legacy as fd
from wavebit.fdtd._prepare.report import report
from wavebit.fdtd._kernel.step import initial_state
from wavebit.fdtd._runtime import runtime
from tests.fixtures.fdtd.fd_helpers import backends, to_numpy


CASES = (
    'vacuum', 'voxel-mixed', 'two-voxel-regions', 'cpml-voxel-mixed',
    'conductive-film', 'conductive-inactive-film', 'lorentz-film',
    'debye-film', 'two-debye-collapsed-film', 'debye-sum-collapsed-film',
    'oscillator-sum-film', 'hybrid-film', 'constitutive-control',
)


@pytest.mark.parametrize('backend', backends())
@pytest.mark.parametrize('complex_fields', (False, True), ids=('real', 'complex'))
@pytest.mark.parametrize('case', CASES)
def test_report_counts_the_initialized_main_state(case, complex_fields, backend, capsys):
    """Actual buffer payload, including inactive reserved and subclass memories.

    Cropped supports and two disjoint regions distinguish compact storage
    from whole-domain or first-term-only estimates. Bloch faces exercise
    the actual complex dtype. Exact collapsed poles keep their declared
    slots; the oracle reads those allocations without inferring activity.
    """
    def array_bytes(tree):
        if isinstance(tree, dict):
            return sum(array_bytes(value) for value in tree.values())
        if isinstance(tree, (tuple, list)):
            return sum(array_bytes(value) for value in tree)
        if tree is None:
            return 0
        return to_numpy(tree).nbytes

    debye = fd.Debye(tau=.2, delta_eps=2.)
    drude = fd.Drude(omega_p=2., gamma=.5)
    lorentz = fd.Lorentz(omega_0=3., gamma=.7, strength=2.)
    inactive = fd.Lorentz(omega_0=4., gamma=1.2, strength=0.)
    mixed = fd.Dispersive(eps_inf=4., poles=(debye, drude, lorentz))
    transverse = fd.Bloch(phase=.31) if complex_fields else fd.Periodic()
    normal = fd.PML(cells=12, wavelength=4., stabilize=0.) if case.startswith('cpml-') else fd.Periodic()
    lengths = (.8, .8, 4. if case.startswith('cpml-') else 2.)
    structures = ()
    if case in ('voxel-mixed', 'cpml-voxel-mixed'):
        structures = (fd.Structure(fd.Region(size=(.3, .4, .5),
                                              center=(-.12, 0., .03)), mixed),)
    elif case == 'two-voxel-regions':
        structures = (
            fd.Structure(fd.Region(size=(.3, .4, .3), center=(-.12, 0., -.4)), mixed),
            fd.Structure(fd.Region(size=(.4, .3, .5), center=(.12, 0., .4)),
                         fd.Dispersive(eps_inf=3., poles=(debye,))),
        )
    elif case == 'constitutive-control':
        # The rotated support and its one-cell halo must clear periodic faces.
        lengths = (1.4, 1.4, 2.)
        structures = (fd.Structure(fd.Box(size=(.43, .47, .54),
                                          center=(.01, -.01, .03), angle=25.),
                                   fd.Material(eps=2.25)),)
    elif case.endswith('-film'):
        materials = {
            'conductive-film': fd.Conductor(eps=4., sigma=2.),
            'conductive-inactive-film': fd.Conductor(eps=4., sigma=0.),
            'lorentz-film': fd.Dispersive(eps_inf=4., poles=(lorentz,)),
            'debye-film': fd.Dispersive(eps_inf=4., poles=(debye,)),
            'two-debye-collapsed-film': fd.Dispersive(eps_inf=4., poles=(
                debye, fd.Debye(tau=.2, delta_eps=0.))),
            'debye-sum-collapsed-film': fd.Dispersive(eps_inf=4., poles=(
                debye, fd.Debye(tau=.2, delta_eps=0.), debye,
                fd.Debye(tau=.2, delta_eps=0.))),
            'oscillator-sum-film': fd.Dispersive(eps_inf=4., poles=(drude, lorentz, inactive)),
            'hybrid-film': fd.Dispersive(eps_inf=4., poles=(debye, drude, inactive)),
        }
        structures = (fd.Structure(fd.Box(size=(1.2, 1.2, .54), center=(0., 0., .03)),
                                   materials[case]),)
    scene = fd.Scene(domain=fd.Domain(size=lengths),
        boundaries=fd.Boundaries(x=transverse, y=fd.Periodic(), z=normal),
        resolution=fd.Resolution(dx=.1), run=fd.Run(steps=3, courant=.2),
        structures=structures)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', fd.CoarseMeshWarning)
        plan = fd.plan(scene)
        rt = runtime(backend)
        params = fd.lower(scene, plan, rt)
        state = initial_state(plan, params, rt)
        public = fd.check(scene)
        supplied_params = report(scene, plan, params)

    assert scene.sources == () and scene.monitors == ()
    assert plan.dtype == ('complex128' if complex_fields else 'float64')
    if case == 'vacuum':
        assert state['mem'] == {} and state['psi'] == {}
    else:
        assert state['mem'], 'the allocation fixture must contain material memory'
    if case == 'constitutive-control':
        assert plan.constitutive, 'the control must allocate constitutive flux memory'
    if case == 'two-voxel-regions':
        assert len(state['mem']) == 2
    if case.startswith('cpml-'):
        assert state['psi'], 'the CPML payload must be present in this control'
    expected = array_bytes({name: state[name] for name in ('E', 'H', 'psi', 'mem')})
    assert public.bytes_per_state == expected
    assert supplied_params.bytes_per_state == expected
    assert 'CPML and material memory' in capsys.readouterr().out
