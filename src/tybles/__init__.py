"""
Data-frame typing / schema documentation module

This module provides helpers to read/process/write `pandas <https://pandas.pydata.org/>`_
:class:`~pandas.DataFrame` instances with simple validation.

.. rubric:: Types

.. py:data:: _RowSpec

    Dataclass specifying a dataframe row
"""

import collections

__version__ = "0.2.0"
import os
from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Literal,
    Mapping,
    NoReturn,
    Sequence,
    Set,
    SupportsIndex,
    TextIO,
    Type,
    TypeVar,
    Union,
    overload,
)

import numpy as np
import pandas as pd
from typing_extensions import ParamSpec, get_type_hints


def _assert_never(x: NoReturn) -> NoReturn:
    raise AssertionError(f"Invalid value: {x!r}")


__all__ = ["Schema", "Tyble", "schema", "tyble"]

_RowSpec = TypeVar("_RowSpec")


@dataclass(frozen=True)
class Schema(Generic[_RowSpec]):
    """
    Describes the structure of a Pandas dataframe

    In Tybles, a schema is derived from a dataclass describing one row of the dataframe.
    """

    #: Row specification
    row_spec: Type[_RowSpec]

    #: Whether to order columns in the dataframe as in the row specification
    order_columns: bool

    #: What to do with missing columns
    #:
    #: This occurs when reading/creating a dataframe *and* when writing/exporting a dataframe.
    #:
    #: The possible values are:
    #:
    #: - "error": raise an error (default)
    #: - "missing": leave the missing columns missing (set ``validate`` to False then)
    #: - "fill": fill the columns with the dtype default value
    missing_columns: Union[Literal["error"], Literal["missing"], Literal["fill"]]

    #: What to do with extra columns present, that are not part of the row specification
    #:
    #: - "drop": remove the extra columns from the dataframe (default)
    #: - "keep": keep the extra columns in the dataframe (note that the dtype is autodetected)
    #: - "error": raise an error
    extra_columns: Union[Literal["drop"], Literal["keep"], Literal["error"]]

    #: Whether to run validation on every row of the data
    #:
    #: If the `typeguard <https://github.com/agronholm/typeguard>`_ library is present, this will
    #: use :func:`typeguard.check_type`, otherwise a simple :func:`isinstance` check will be done.
    validate: bool

    #: Names of the fields in the schema, in order of definition
    field_names: Sequence[str]

    #: Mapping of field names with associated dtypes
    #:
    #: Can also serve as a ``dtype=`` argument for various Pandas functions
    dtypes: Mapping[str, np.dtype]

    #: Mapping of field names with associated annotated types
    annotated_types: Mapping[str, type]

    def validate_row(self, row: _RowSpec) -> None:
        """
        Validates the given row and raises an exception if validation fails

        Args:
            row: Row to validate

        Raises:
            TypeError: If typeguard or standard validation failed
            BeartypeException: If beartype failed
        """
        checked = False
        try:
            from typeguard import check_type

            check_type("Row in dataframe", row, self.row_spec)
            checked = True
        except ImportError:
            pass

        try:
            from beartype.abby import die_if_unbearable

            for name, typ in self.annotated_types.items():
                die_if_unbearable(getattr(row, name), typ)
            checked = True
        except ImportError:
            pass

        if not checked:
            for name, typ in self.annotated_types.items():
                value = getattr(row, name)
                if not isinstance(value, typ):
                    raise TypeError(
                        f"Field {name} with value {value} does not conform to type {typ}"
                    )

    @overload
    def from_rows(
        self, rows: Sequence[_RowSpec], return_type: Literal["DataFrame"] = "DataFrame", **kwargs
    ) -> pd.DataFrame:
        pass

    @overload
    def from_rows(
        self, rows: Sequence[_RowSpec], return_type: Literal["Tyble"], **kwargs
    ) -> pd.DataFrame:
        pass

    def from_rows(
        self,
        rows: Sequence[_RowSpec],
        return_type: Union[Literal["DataFrame"], Literal["Tyble"]] = "DataFrame",
        **kwargs,
    ) -> Union[pd.DataFrame, "Tyble[_RowSpec]"]:
        """
        Returns a pandas DataFrame (possibly as an enriched Tyble) from row instances

        Args:
            rows: Rows as a sequence of dataclass instances

        Keyword Args:
            return_type: Whether to return a pandas :class:`~pandas.DataFrame` (default)
                        or a :class:`.Tyble` instance
            kwargs: Extra keyword arguments are passed to :meth:`pandas.DataFrame.from_records`

        Returns:
            A pandas DataFrame, possibly wrapped in a Tyble
        """
        df = pd.DataFrame.from_records([asdict(row) for row in rows])
        return self.process_raw_data_frame(df, return_type)

    @overload
    def read_csv(
        self,
        filepath_or_buffer: Union[TextIO, str, bytes, os.PathLike],
        return_type: Literal["DataFrame"] = "DataFrame",
        **kw_args,
    ) -> pd.DataFrame:
        pass

    @overload
    def read_csv(
        self,
        filepath_or_buffer: Union[TextIO, str, bytes, os.PathLike],
        return_type: Literal["Tyble"],
        **kw_args,
    ) -> "Tyble[_RowSpec]":
        pass

    def read_csv(
        self,
        filepath_or_buffer: Union[TextIO, str, bytes, os.PathLike],
        return_type: Union[Literal["DataFrame"], Literal["Tyble"]] = "DataFrame",
        **kw_args,
    ) -> Union[pd.DataFrame, "Tyble[_RowSpec]"]:
        """
        Reads a pandas DataFrame from a CSV file, shaping up and validating the data on demand

        Args:
            filepath_or_buffer: Path or open file to read from

        Keyword Args:
            return_type: Whether to return a pandas :class:`~pandas.DataFrame` (default)
                        or a :class:`.Tyble` instance
            kw_args: Additional keyword arguments not listed above are passed to :func:`pandas.read_csv`

        Returns:
            A pandas dataframe, possibly wrapped in a Tyble
        """
        return self.process_raw_data_frame(
            pd.read_csv(
                filepath_or_buffer,
                dtype=self.dtypes,
                **kw_args,
            ),
            return_type,
        )

    @overload
    def process_raw_data_frame(
        self, df: pd.DataFrame, return_type: Literal["DataFrame"] = "DataFrame"
    ) -> pd.DataFrame:
        pass

    @overload
    def process_raw_data_frame(
        self, df: pd.DataFrame, return_type: Literal["Tyble"]
    ) -> "Tyble[_RowSpec]":
        pass

    def process_raw_data_frame(
        self,
        df: pd.DataFrame,
        return_type: Union[Literal["DataFrame"], Literal["Tyble"]] = "DataFrame",
    ) -> Union[pd.DataFrame, "Tyble[_RowSpec]"]:
        """
        Args:
            df: Dataframe to process, will be mutated.

                In any case, one should use the dataframe returned by this function.
                (The code may or may not mutate in place this given dataframe.)

        Raises:
            ValueError: If the dataframe fails the ``missing_columns`` or ``extra_columns`` checks
            TypeError: If typeguard validation failed
            BeartypeException: If beartype failed

        Returns:
            The processed dataframe or a dataframe wrapped in a :class:`.Tyble` instance
        """
        if self.missing_columns != "missing":
            missing: Set[str] = set(self.field_names).difference(df.columns)
            if missing:
                if self.missing_columns == "error":
                    raise ValueError("Missing columns in CSV file: " + ", ".join(missing))
                if self.missing_columns == "fill":
                    for name, dt in self.dtypes.items():
                        if name in missing:
                            df[name] = pd.Series(np.zeros((len(df),), dtype=dt), index=df.index)
        if self.extra_columns != "keep":
            extra: Set[str] = set(df.columns).difference(self.field_names)
            if extra:
                if self.extra_columns == "error":
                    raise ValueError("Extra columns in CSV file: " + ", ".join(extra))
                if self.extra_columns == "drop":
                    df.drop(list(extra), axis="columns", inplace=True)
        if self.order_columns:
            in_spec: Sequence[str] = [n for n in self.field_names if n in df.columns]
            extra1: Sequence[str] = [n for n in df.columns if n not in self.field_names]
            df = df.loc[:, [*in_spec, *extra1]]
        if self.validate:
            for row in Tyble(df, self):
                self.validate_row(row)
        if return_type == "DataFrame":
            return df
        elif return_type == "Tyble":
            return Tyble(df, self)
        else:
            _assert_never(return_type)


def schema(
    row_spec: Type[_RowSpec],
    *,
    order_columns: bool = True,
    missing_columns: Union[Literal["error"], Literal["missing"], Literal["fill"]] = "error",
    extra_columns: Union[Literal["drop"], Literal["keep"], Literal["error"]] = "drop",
    validate: bool = True,
) -> Schema[_RowSpec]:
    """
    Creates a dataframe schema from a row specification dataclass

    For detailed description of the keyword arguments, look up the :class:`.Schema` attributes
    documentation.

    Args:
        row_spec: Data class specifying a dataframe row

    Keyword Args:
        order_columns: Whether to order the dataframe columns as in the specification
        missing_columns: What to do with missing columns
        extra_columns: What to do with extra columns
        validate: Whether to perform validation

    Returns:
        A dataframe schema
    """
    assert is_dataclass(row_spec), "The row specification must be a dataclass"
    assert hasattr(row_spec, "__annotations__")

    if missing_columns == "missing":
        assert not validate, "When missing_columns = missing, validate should be False"

    dtypes = {n: np.dtype(t) for n, t in get_type_hints(row_spec, include_extras=False).items()}

    return Schema(
        row_spec,
        dtypes=dtypes,
        annotated_types=get_type_hints(row_spec, include_extras=True),
        field_names=[f.name for f in fields(row_spec)],
        order_columns=order_columns,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        validate=validate,
    )


@dataclass(frozen=True)
class Tyble(collections.Sequence[_RowSpec], Generic[_RowSpec]):
    """
    Describes a Pandas dataframe enriched with a schema
    """

    data_frame: pd.DataFrame
    schema: Schema[_RowSpec]

    def __len__(self) -> int:
        return len(self.data_frame)

    def to_rows(self) -> Sequence[_RowSpec]:
        return self[:]

    @overload
    def __getitem__(self, index: int) -> _RowSpec:
        pass

    @overload
    def __getitem__(self, index: slice) -> Sequence[_RowSpec]:
        pass

    @overload
    def __getitem__(self, index: Sequence[int]) -> Sequence[_RowSpec]:
        pass

    @overload
    def __getitem__(self, index: SupportsIndex) -> _RowSpec:
        pass

    @overload
    def __getitem__(self, index: Iterable[SupportsIndex]) -> Sequence[_RowSpec]:
        pass

    def __getitem__(self, index: Any) -> Union[_RowSpec, Sequence[_RowSpec]]:
        if isinstance(index, SupportsIndex):
            row = self.data_frame.iloc[index.__index__()]
            content = {name: row[name] for name in self.schema.dtypes.keys()}
            return self.schema.row_spec(**content)
        elif isinstance(index, Iterable):
            return [self[i] for i in index]
        elif isinstance(index, slice):
            return self[range(*index.indices(len(self.data_frame)))]
        else:
            raise ValueError("Argument must be either an index or a sequence of indices")

    def __repr__(self) -> str:
        tn = self.schema.row_spec.__name__
        if len(self) == 0:
            return f"Empty Tyble for row_spec={tn}"
        else:
            return "\n".join(
                [
                    f"Tyble: self.schema.row_spec={tn}",
                    f"       self[0]={self[0]}",
                    "       self.data_frame=",
                    *self.data_frame.__repr__().split("\n"),
                ]
            )


def tyble(data_frame: pd.DataFrame, schema: Schema[_RowSpec]) -> Tyble[_RowSpec]:
    return Tyble(data_frame, schema)
