from ast import Assert
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
class RowSpec:
    a: np.float64
    b: np.float64
    c: np.float64


@dataclass(frozen=True)
class ValidatedRowSpec:
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

    tb.read_csv(StringIO(csv_error), ValidatedRowSpec, validate=False)
    tb.read_csv(StringIO(csv_error), RowSpec, validate=True)
    tb.read_csv(StringIO(csv), ValidatedRowSpec, validate=True)
    with pytest.raises(AssertionError):
        tb.read_csv(StringIO(csv_error), ValidatedRowSpec, validate=True)


def test_type_error() -> None:
    csv = """
a,c,b
x,2,3
y,5,6
""".strip()
    with pytest.raises(ValueError):
        tb.read_csv(StringIO(csv), RowSpec)


def test_extra_columns() -> None:

    csv = """
a,c,b,d
1,2,3,4
4,5,6,7
""".strip()

    with pytest.raises(ValueError):
        tb.read_csv(StringIO(csv), RowSpec, extra_columns="error")

    df1 = tb.read_csv(StringIO(csv), RowSpec, extra_columns="drop")
    df2 = tb.read_csv(StringIO(csv), RowSpec, extra_columns="keep")
    assert list(df1.columns) == ["a", "b", "c"]
    assert list(df2.columns) == ["a", "b", "c", "d"]


def test_missing_columns() -> None:
    csv = """
a,c
1,2
4,5
""".strip()

    with pytest.raises(ValueError):
        tb.read_csv(StringIO(csv), RowSpec, missing_columns="error")

    df1 = tb.read_csv(StringIO(csv), RowSpec, missing_columns="fill")
    df2 = tb.read_csv(StringIO(csv), RowSpec, missing_columns="missing", validate=False)
    assert list(df1.columns) == ["a", "b", "c"]
    assert list(df2.columns) == ["a", "c"]


def test_order_columns() -> None:

    csv = """
a,c,b
1,2,3
4,5,6
""".strip()

    df1 = tb.read_csv(StringIO(csv), RowSpec, order_columns=True)
    df2 = tb.read_csv(StringIO(csv), RowSpec, order_columns=False)
    assert list(df1.columns) == ["a", "b", "c"]
    assert list(df2.columns) == ["a", "c", "b"]
