"""Maintained study receipts must hash the implementation after a source move."""
import hashlib
import json
from pathlib import Path

import pytest


def test_angular_resolution_header_identifies_every_conforming_source(monkeypatch, capsys):
    from studies.rcwa.nvf_pair_angular_resolution import nvf_pair_angular_resolution as study

    class HeaderRecorded(Exception):
        pass

    def stop_before_calculation(*args, **kwargs):
        raise HeaderRecorded

    monkeypatch.setattr(study, 'CartesianPairCornerGalerkinGapCell', stop_before_calculation)
    monkeypatch.setattr('sys.argv', [str(Path(study.__file__))])
    with pytest.raises(HeaderRecorded):
        study.main()
    header = json.loads(capsys.readouterr().out)
    root = Path(__file__).resolve().parents[2]
    sources = list((root / 'src/wavebit/rcwa/_engine/conforming').glob('*.py'))
    assert len(sources) > 10
    expected = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sources}
    assert expected.items() <= header['source_sha256'].items()


@pytest.mark.parametrize('relative,names,table', [
    ('studies/cross_solver/junction/fdtd_junction_mutations.py',
     {'J', 'mutants'}, 'mutants'),
    ('studies/cross_solver/metasurface/fdtd_metasurface_mutations.py',
     {'TRANSLATION', 'ROWS', 'CONICAL', 'MUTATIONS'}, 'MUTATIONS'),
])
def test_cross_solver_mutations_still_target_one_existing_source_anchor(relative, names, table):
    # Load only the declaration table: importing the historical junction driver
    # would execute mutations. This preflight never writes a solver source file.
    import ast
    root = Path(__file__).resolve().parents[2]
    tree = ast.parse((root / relative).read_text())
    declarations = [node for node in tree.body if isinstance(node, ast.Assign)
                    and any(isinstance(target, ast.Name) and target.id in names
                            for target in node.targets)]
    namespace = {}
    exec(compile(ast.Module(body=declarations, type_ignores=[]), relative, 'exec'), namespace)
    rows = namespace[table]
    assert rows
    for row in rows:
        changes = row[1] if table == 'mutants' else [
            (str(Path('src/wavebit/fdtd') / row[1]), row[2], row[3])]
        for source, before, after in changes:
            target = root / source
            assert target.is_file(), (row[0], source)
            assert target.read_text().count(before) == 1, (row[0], source)
            assert before != after
