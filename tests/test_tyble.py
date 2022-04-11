from dataclasses import dataclass
from io import StringIO

import numpy as np
import tybles as tb


@dataclass(frozen=True)
class Row:
    a: np.float64
    b: np.float64
    c: np.float64


def test_tyble() -> None:
    csv = """
a,c,b
1,2,3
2,5,6
""".strip()
    schema = tb.schema(Row)
    tyble = schema.read_csv(StringIO(csv), "Tyble")
    assert tyble[0] == Row(np.float64(1), np.float64(3), np.float64(2))
    assert len(tyble) == 2
    assert [float(row.a) for row in tyble] == [1.0, 2.0]
