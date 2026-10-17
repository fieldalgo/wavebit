"""Optional FDTD checkpoint policy preserves old v1 envelopes and job keys."""
import json

import pytest

from wavebit import Execution, Job, Units, fdtd


def job():
    problem=fdtd.Problem(units=Units(length='um'),scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.,1.,1.)),boundaries=fdtd.Boundaries.periodic(z=fdtd.Open())))
    numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),duration=fdtd.Duration(steps=3))
    return Job(solver='fdtd',problem=problem,numerics=numerics)


def test_checkpoint_default_preserves_original_v1_job_key_and_omission():
    # Frozen before optional Numerics additions, also pinned by the memory-bound gate.
    expected='78499379496ea489cd43a4a40f5d29fb07400a955e722a5507f30d36ed707e0e'
    original=job()
    assert original.key==expected
    fields=json.loads(original.to_json())['numerics']['fields']
    assert 'checkpoint_steps' not in fields
    restored=Job.from_json(original.to_json())
    assert restored.numerics.checkpoint_steps is None
    assert restored.key==expected
    assert original.replace(numerics=original.numerics.replace(checkpoint_steps=None)).to_json()==original.to_json()


@pytest.mark.parametrize('block',[1,11,200])
def test_explicit_checkpoint_roundtrip_retains_policy_and_changes_key(block):
    original=job().replace(execution=Execution(backend='torch',device='cuda:0'))
    scheduled=original.replace(numerics=original.numerics.replace(checkpoint_steps=block))
    restored=Job.from_json(scheduled.to_json())
    assert restored.numerics.checkpoint_steps==block
    assert restored.execution==original.execution
    assert restored.key==scheduled.key!=original.key
    assert restored.replace(numerics=restored.numerics.replace(checkpoint_steps=None)).key==original.key
