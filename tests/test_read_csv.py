from dataclasses import dataclass
from io import StringIO

import beartype
import beartype.abby
import beartype.roar
import beartype.vale
import numpy as np
import pytest
import tybles as tb
from typing_extensions import Annotated


@dataclass(frozen=True)
class Row:
    a: np.float64
    b: np.float64
    c: np.float64


@dataclass(frozen=True)
class NonNegRow:
    a: np.float64
    b: np.float64
    c: Annotated[np.float64, beartype.vale.Is[lambda x: x >= 0]]


def test_validate() -> None:
    csv = """
a,c,b
1,2,3
2,5,6
""".strip()
    csv_error = """
a,c,b
1,2,3
2,-5,6
""".strip()
    tb.schema(NonNegRow, validate=False).read_csv(StringIO(csv_error))
    tb.schema(Row, validate=True).read_csv(StringIO(csv_error))
    tb.schema(NonNegRow, validate=True).read_csv(StringIO(csv))
    with pytest.raises(AssertionError):
        tb.schema(NonNegRow, validate=True).read_csv(StringIO(csv_error))


def test_type_error() -> None:
    csv = """
a,c,b
x,2,3
y,5,6
""".strip()
    with pytest.raises(ValueError):
        tb.schema(Row).read_csv(StringIO(csv))


def test_extra_columns() -> None:

    csv = """
a,c,b,d
1,2,3,4
4,5,6,7
""".strip()

    with pytest.raises(ValueError):
        tb.schema(Row, extra_columns="error").read_csv(StringIO(csv))

    df1 = tb.schema(Row, extra_columns="drop").read_csv(StringIO(csv))
    df2 = tb.schema(Row, extra_columns="keep").read_csv(StringIO(csv))
    assert list(df1.columns) == ["a", "b", "c"]
    assert list(df2.columns) == ["a", "b", "c", "d"]


def test_missing_columns() -> None:
    csv = """
a,c
1,2
4,5
""".strip()

    with pytest.raises(ValueError):
        tb.schema(Row, missing_columns="error").read_csv(StringIO(csv))

    df1 = tb.schema(Row, missing_columns="fill").read_csv(StringIO(csv))
    df2 = tb.schema(Row, missing_columns="missing", validate=False).read_csv(StringIO(csv))
    assert list(df1.columns) == ["a", "b", "c"]
    assert list(df2.columns) == ["a", "c"]


def test_order_columns() -> None:

    csv = """
a,c,b
1,2,3
4,5,6
""".strip()

    df1 = tb.schema(Row, order_columns=True).read_csv(StringIO(csv))
    df2 = tb.schema(Row, order_columns=False).read_csv(StringIO(csv))
    assert list(df1.columns) == ["a", "b", "c"]
    assert list(df2.columns) == ["a", "c", "b"]
