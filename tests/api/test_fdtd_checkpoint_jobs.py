"""An optional FDTD checkpoint policy leaves a job that does not use it unchanged."""
import json

import pytest

from wavebit import Execution, Job, Units, fdtd


def job():
    problem=fdtd.Problem(units=Units(length='um'),scene=fdtd.Scene(
        domain=fdtd.Domain(size=(1.,1.,1.)),boundaries=fdtd.Boundaries.periodic(z=fdtd.Open())))
    numerics=fdtd.Numerics(grid=fdtd.UniformGrid(spacing=.05),duration=fdtd.Duration(steps=3))
    return Job(solver='fdtd',problem=problem,numerics=numerics)


def test_a_default_checkpoint_policy_is_not_written_to_a_job():
    """Written only off its default, so a job that does not use it keeps the key it had without
    the field. The key itself was pinned in wavebit.job/v1; v2 changed every key."""
    original=job()
    assert 'checkpoint_steps' in fdtd.Numerics._FIELDS_OMITTED_AT_DEFAULT
    fields=json.loads(original.to_json())['numerics']['fields']
    assert 'checkpoint_steps' not in fields
    restored=Job.from_json(original.to_json())
    assert restored.numerics.checkpoint_steps is None
    assert restored.key==original.key
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
