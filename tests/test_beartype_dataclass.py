from dataclasses import dataclass

import beartype
import beartype.abby
import beartype.roar
import beartype.vale
import pytest
from typing_extensions import Annotated, get_type_hints


@beartype.beartype
@dataclass(frozen=True)
class WithDecorator:
    a: Annotated[int, beartype.vale.Is[lambda x: x >= 0]]


@dataclass(frozen=True)
class WithoutDecorator:
    a: Annotated[int, beartype.vale.Is[lambda x: x >= 0]]


def test_beartype_decorator() -> None:
    with pytest.raises(beartype.roar.BeartypeException):
        WithDecorator(-1)


@pytest.mark.skip(reason="currently fails")
def test_unbearable_dataclass() -> None:
    with pytest.raises(beartype.roar.BeartypeException):
        data = WithoutDecorator(-1)
        beartype.abby.die_if_unbearable(data, WithoutDecorator)


@pytest.mark.skip(reason="currently fails")
def test_dataclass_in_argument() -> None:
    with pytest.raises(beartype.roar.BeartypeException):

        @beartype.beartype
        def fun(d: WithoutDecorator) -> None:
            pass

        data = WithoutDecorator(-1)
        fun(data)


def test_unbearable_field() -> None:
    hints = get_type_hints(WithoutDecorator, include_extras=True)
    with pytest.raises(beartype.roar.BeartypeException):
        data = WithoutDecorator(-1)
        beartype.abby.die_if_unbearable(data.a, hints["a"])
