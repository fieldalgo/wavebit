"""JobSpec is the serialisable form of a run, and ``fieldwave.run`` executes it statelessly.

Two properties carry the whole design and are gated here:

* **Parity** — a spec must produce byte-identical results to the equivalent
  :func:`fieldwave.rcwa` call. The spec layer is a new way to *say* a run, never a new way
  to *compute* one.
* **Round-trip** — a spec must survive JSON unchanged, because it exists to cross process
  and network boundaries. Equal specs must hash equal, and differing ones must not.

Validation is tested for rejection, not just acceptance: a bad spec caught at submission is
far cheaper than one caught on a worker.
"""
import json

import numpy as np
import pytest

import fieldwave
from fieldwave import JobSpec, Sweep, SpecError, Backend, Pol, Factorization


BINARY_GRATING = {
    'numerics': {'backend': 'numpy', 'fto': [10, 0]},
    'stack': {
        'n_top': 1.0, 'n_bot': 1.5, 'period': [700.0],
        'layers': [{'thickness': 300.0, 'eps': [[1.0, 1.0, 3.5, 3.5]]}],
    },
    'incidence': {'wavelength': 900.0, 'theta': 0.0, 'phi': 0.0, 'pol': 'te'},
}


# ---------------------------------------------------------------- parity

def test_run_matches_the_equivalent_rcwa_call():
    """The gate: a spec computes exactly what the legacy keyword call computes."""
    result = fieldwave.run(BINARY_GRATING)

    reference = fieldwave.rcwa(
        backend=0, wavelength=900., period=[700.], fto=[10, 0],
        n_top=1., n_bot=1.5, theta=0., phi=0., pol=0.,
        ucell=np.array([[[1.0, 1.0, 3.5, 3.5]]]), thickness=[300.],
    ).conv_solve()

    assert np.array_equal(result.de_ri, reference.de_ri)
    assert np.array_equal(result.de_ti, reference.de_ti)
    assert result.de_ri.sum() + result.de_ti.sum() == pytest.approx(1.0, abs=1e-13)


def test_run_accepts_spec_object_mapping_and_json():
    """All three input forms are the same run — the caller picks whichever it holds."""
    spec = JobSpec.from_dict(BINARY_GRATING)
    from_obj = fieldwave.run(spec)
    from_map = fieldwave.run(BINARY_GRATING)
    from_str = fieldwave.run(spec.to_json())

    for other in (from_map, from_str):
        assert np.array_equal(from_obj.de_ri, other.de_ri)
        assert np.array_equal(from_obj.de_ti, other.de_ti)


# ---------------------------------------------------------------- round-trip and identity

def test_spec_survives_json_unchanged():
    spec = JobSpec.from_dict(BINARY_GRATING)
    revived = JobSpec.from_json(spec.to_json())

    assert revived.to_dict() == spec.to_dict()
    assert revived.key() == spec.key()
    json.dumps(spec.to_dict())                      # must be plain JSON, not just dict-shaped


def test_key_tracks_content_not_spelling():
    spec = JobSpec.from_dict(BINARY_GRATING)

    # same run, written differently: enum by name vs by int, defaults stated vs omitted
    reworded = dict(BINARY_GRATING)
    reworded['numerics'] = {'backend': 0, 'fto': [10, 0], 'type_complex': 128}
    reworded['incidence'] = {'wavelength': 900.0, 'theta': 0.0, 'phi': 0.0, 'pol': 0}
    assert JobSpec.from_dict(reworded).key() == spec.key()

    # a different run
    changed = dict(BINARY_GRATING)
    changed['incidence'] = {**BINARY_GRATING['incidence'], 'wavelength': 901.0}
    assert JobSpec.from_dict(changed).key() != spec.key()


def test_pol_normalises_to_a_number_and_keeps_fractions():
    """``pol`` is the TM fraction, so names are sugar over a continuum — not a two-valued flag."""
    assert JobSpec.from_dict(BINARY_GRATING).to_dict()['incidence']['pol'] == 0
    tm = JobSpec.from_dict({**BINARY_GRATING,
                            'incidence': {**BINARY_GRATING['incidence'], 'pol': 'tm'}})
    assert tm.to_dict()['incidence']['pol'] == 1

    half = JobSpec.from_dict({**BINARY_GRATING,
                              'incidence': {**BINARY_GRATING['incidence'], 'pol': 0.5}})
    assert half.to_dict()['incidence']['pol'] == 0.5
    assert JobSpec.from_json(half.to_json()).key() == half.key()
    assert half.key() != tm.key()


def test_enums_accept_name_or_value():
    spec = JobSpec.from_dict({
        **BINARY_GRATING,
        'numerics': {'backend': 'JAX', 'factorization': 'nvf', 'fto': 3},
    })
    assert spec.numerics.backend is Backend.JAX
    assert spec.numerics.factorization is Factorization.NVF
    assert spec.numerics.fto == (3, 0)              # a bare int is the x order


# ---------------------------------------------------------------- sweeps

def test_sweep_expands_to_the_cartesian_product():
    spec = JobSpec.from_dict({
        **BINARY_GRATING,
        # radians, and the test said 10.0 until the units rule caught it
        'sweep': {'wavelength': {'linspace': [500.0, 900.0, 5]}, 'theta': [0.0, 0.2]},
    })
    assert len(spec) == 10                          # 5 wavelengths x 2 angles

    points = spec.expand()
    assert len(points) == 10
    assert [(p.incidence.wavelength, p.incidence.theta) for p in points[:3]] == [
        (500.0, 0.0), (500.0, 0.2), (600.0, 0.0)]

    # every point is independently executable, and distinct
    assert all(not p.sweep for p in points)
    assert len({p.key() for p in points}) == 10
    # the scene is untouched by the sweep
    assert all(p.stack.thickness == spec.stack.thickness for p in points)


def test_expand_is_identity_without_a_sweep():
    spec = JobSpec.from_dict(BINARY_GRATING)
    assert len(spec) == 1
    assert spec.expand() == [spec]


def test_linspace_endpoints_are_exact():
    axis = Sweep.from_dict({'wavelength': {'linspace': [500.0, 1000.0, 6]}}).axes['wavelength']
    assert axis[0] == 500.0
    assert axis[-1] == pytest.approx(1000.0)
    assert len(axis) == 6
    assert Sweep.from_dict({'theta': {'linspace': [0.7, 0.7, 1]}}).axes['theta'] == [0.7]


def test_running_a_sweep_spec_is_refused_with_the_way_out():
    spec = JobSpec.from_dict({**BINARY_GRATING, 'sweep': {'wavelength': [500.0, 600.0]}})
    with pytest.raises(SpecError, match=r'expand'):
        fieldwave.run(spec)


def test_expanded_sweep_points_match_direct_calls():
    """Expansion must not perturb the run — point i is exactly the run with that wavelength."""
    spec = JobSpec.from_dict({**BINARY_GRATING, 'sweep': {'wavelength': [800.0, 900.0]}})

    for point in spec.expand():
        direct = dict(BINARY_GRATING)
        direct['incidence'] = {**BINARY_GRATING['incidence'],
                               'wavelength': point.incidence.wavelength}
        assert np.array_equal(fieldwave.run(point).de_ri, fieldwave.run(direct).de_ri)


# ---------------------------------------------------------------- rejection

@pytest.mark.parametrize('spec, message', [
    ({**BINARY_GRATING, 'wavelength': 900.0}, r'unknown field'),          # misplaced key
    ({**BINARY_GRATING, 'numerics': {'fto': [10, 0], 'backand': 0}}, r'unknown field'),
    ({**BINARY_GRATING, 'numerics': {'backend': 'cupy'}}, r'not a Backend'),
    ({**BINARY_GRATING, 'numerics': {'fto': [1, 2, 3]}}, r'two non-negative ints'),
    ({**BINARY_GRATING, 'numerics': {'fto': [-1, 0]}}, r'two non-negative ints'),
    ({**BINARY_GRATING, 'numerics': {'type_complex': 32}}, r'128 or 64'),
    ({**BINARY_GRATING, 'version': '99'}, r'version'),
    ({**BINARY_GRATING, 'solver': 'fdtd'}, r'only .rcwa.'),
    ({**BINARY_GRATING, 'outputs': []}, r'no outputs'),
    ({**BINARY_GRATING, 'sweep': {'thickness': [1.0]}}, r'unknown field'),
    ({**BINARY_GRATING, 'sweep': {'wavelength': {}}}, r"one of 'values' or 'linspace'"),
    ({**BINARY_GRATING, 'sweep': {'wavelength': {'values': [], }}}, r'no values'),
    ({**BINARY_GRATING, 'sweep': {'wavelength': {'linspace': [1, 2]}}}, r'start, stop, num'),
    ({**BINARY_GRATING,
      'stack': {'layers': [{'thickness': -1.0, 'eps': [[1.0]]}]}}, r'non-negative'),
    ({**BINARY_GRATING, 'stack': {'layers': [{'eps': [[1.0]]}]}}, r"'thickness' and 'eps'"),
])
def test_bad_specs_are_rejected_at_submission(spec, message):
    with pytest.raises(SpecError, match=message):
        JobSpec.from_dict(spec)


def test_raster_ref_is_refused_by_the_stateless_entry_point():
    """A reference is not self-contained; ``run`` takes specs that fully determine a run."""
    spec = {**BINARY_GRATING,
            'stack': {'layers': [{'thickness': 300.0, 'eps': {'raster_ref': 's3://x'}}]}}
    with pytest.raises(SpecError, match=r'raster_ref'):
        JobSpec.from_dict(spec)


def test_run_rejects_a_type_it_cannot_interpret():
    with pytest.raises(SpecError, match=r'expected a JobSpec'):
        fieldwave.run(42)


# ---------------------------------------------------------------- units

def test_degrees_where_radians_belong_are_refused():
    """The trap this rule exists for: nothing downstream can catch it.

    A polar angle of 10 makes the solver return finite numbers that conserve no energy, and
    they look like absorption — which the solver's physical check deliberately permits,
    since a lossy stack does exactly that. So the only place it can be caught is here.
    """
    with pytest.raises(SpecError, match=r'RADIANS'):
        JobSpec.from_dict({**BINARY_GRATING,
                           'incidence': {**BINARY_GRATING['incidence'], 'theta': 10.0}})

    # and by the same route a sweep takes, which reaches the solver through expand()
    with pytest.raises(SpecError, match=r'RADIANS'):
        JobSpec.from_dict({**BINARY_GRATING, 'sweep': {'theta': [0.0, 5.0]}})


def test_real_polar_angles_pass():
    import math
    spec = JobSpec.from_dict({
        **BINARY_GRATING,
        'sweep': {'theta': [0.0, math.radians(10), math.radians(60), -math.radians(45)]}})
    assert len(spec) == 4
    # A lossless stack must still conserve energy at a real angle.
    for point in spec.expand():
        r = fieldwave.run(point)
        assert r.de_ri.sum() + r.de_ti.sum() == pytest.approx(1.0, abs=1e-9)
