"""
Data-frame typing / schema documentation module

This module provides helpers to read/process/write `pandas <https://pandas.pydata.org/>`_
:class:`~pandas.DataFrame` instances with simple validation.
"""
from __future__ import annotations

__version__ = "0.2.0"
import os
from dataclasses import dataclass, fields
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterable,
    Literal,
    Mapping,
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

__all__ = ["Schema", "Rows", "read_csv"]
#: Dataclass type that specifies rows
#:
#: Must be a type with __annotations__ set, which is the case when one applies PEP 484 type
#: hints.
_RowSpec = TypeVar("_RowSpec")

#: Parameter specification for wrapped Pandas functions
_Params = ParamSpec("_Params")

_MissingColumns = Union[Literal["error"], Literal["missing"], Literal["fill"]]

_ExtraColumns = Union[Literal["drop"], Literal["keep"], Literal["error"]]


@dataclass(frozen=True)
class Schema(Generic[_RowSpec]):
    """
    Describes the structure of a Pandas dataframe

    In Tybles, a schema is derived from a dataclass describing one row of the dataframe.
    """

    #: Row specification
    row_spec: Type[_RowSpec]

    #: Names of the fields in the schema, in order of definition
    field_names: Sequence[str]

    #: Mapping of field names with associated dtypes
    #:
    #: Can also serve as a ``dtype=`` argument for various Pandas functions
    dtypes: Mapping[str, np.dtype]

    #: Mapping of field names with associated annotated types
    annotated_types: Mapping[str, type]

    def validate(self, row: _RowSpec) -> None:
        try:
            from beartype.abby import die_if_unbearable

            for name, typ in self.annotated_types.items():
                die_if_unbearable(getattr(row, name), typ)
        except ImportError:
            for name, typ in self.annotated_types.items():
                assert isinstance(getattr(row, name), typ)

    @staticmethod
    def make(row_spec: Type[_RowSpec]) -> Schema[_RowSpec]:
        """
        Creates a schema from a dataclass
        """

        assert hasattr(row_spec, "__annotations__")
        fn = [f.name for f in fields(row_spec)]
        f = {n: np.dtype(t) for n, t in get_type_hints(row_spec, include_extras=False).items()}
        at = get_type_hints(row_spec, include_extras=True)
        return Schema(row_spec=row_spec, dtypes=f, annotated_types=at, field_names=fn)

    def wrap(
        self,
        f: Callable[_Params, pd.DataFrame],
        order_columns: bool = True,
        missing_columns: _MissingColumns = "error",
        extra_columns: _ExtraColumns = "drop",
        validate: bool = True,
    ) -> Callable[_Params, pd.DataFrame]:
        """
        Wraps a dataframe construction function so that the dataframe is validated/post-processed

        Args:
            f: Pandas function to wrap (e.g. :func:`pandas.read_csv`)
            order_columns: Whether to order columns in the dataframe as in the specification
            missing_columns: What to do with missing columns in the dataframe:

                - "error": raise an error (default)
                - "missing": leave the missing columns missing (set ``validate`` to False then)
                - "fill": fill the columns with the dtype default value

            extra_columns: What to do with extra columns present in the dataframe

                - "drop": remove the extra columns from the dataframe (default)
                - "keep": keep the extra columns in the dataframe
                  (note that the dtype is autodetected)
                - "error": raise an error

            validate: Whether to run validation on every row
                (uses beartype if the `beartype <https://github.com/beartype/beartype>`_
                otherwise runs simple :func:`isinstance` checks)

        Returns:
            The wrapped function
        """

        def wrapped(*args: _Params.args, **kwargs: _Params.kwargs) -> pd.DataFrame:
            df: pd.DataFrame = f(*args, **kwargs, dtype=self.dtypes)  # type: ignore
            df = self.process_raw_data_frame(
                df,
                missing_columns=missing_columns,
                extra_columns=extra_columns,
                validate=validate,
                order_columns=order_columns,
            )
            return df

        return wrapped

    def process_raw_data_frame(
        self,
        df: pd.DataFrame,
        order_columns: bool = True,
        missing_columns: Union[Literal["error"], Literal["missing"], Literal["fill"]] = "error",
        extra_columns: Union[Literal["drop"], Literal["keep"], Literal["error"]] = "drop",
        validate: bool = True,
    ) -> pd.DataFrame:
        """
        Args:
            df: Dataframe to process, will be mutated.

                In any case, one should use the dataframe returned by this function.
                (The code may or may not mutate in place this given dataframe.)
            order_columns: Whether to order columns in the dataframe as in the specification
            missing_columns: What to do with missing columns in the dataframe:

                - "error": raise an error (default)
                - "missing": leave the missing columns missing (set ``validate`` to False then)
                - "fill": fill the columns with the dtype default value

            extra_columns: What to do with extra columns present in the dataframe:

                - "drop": remove the extra columns from the dataframe (default)
                - "keep": keep the extra columns in the dataframe (note that the dtype is autodetected)
                - "error": raise an error

            validate: Whether to run validation on every row
                    (uses beartype if the `beartype <https://github.com/beartype/beartype>`_
                    otherwise runs simple :func:`isinstance` checks)

        Raises:
            ValueError: If the dataframe fails the ``missing_columns`` or ``extra_columns`` checks
            BeartypeException: If beartype validation has been requested and failed

        Returns:
            The processed dataframe
        """
        if missing_columns != "missing":
            missing: Set[str] = set(self.field_names).difference(df.columns)
            if missing:
                if missing_columns == "error":
                    raise ValueError("Missing columns in CSV file: " + ", ".join(missing))
                if missing_columns == "fill":
                    for name, dt in self.dtypes.items():
                        if name in missing:
                            df[name] = pd.Series(np.zeros((len(df),), dtype=dt), index=df.index)
        if extra_columns != "keep":
            extra: Set[str] = set(df.columns).difference(self.field_names)
            if extra:
                if extra_columns == "error":
                    raise ValueError("Extra columns in CSV file: " + ", ".join(extra))
                if extra_columns == "drop":
                    df.drop(list(extra), axis="columns", inplace=True)
        if order_columns:
            in_spec: Sequence[str] = [n for n in self.field_names if n in df.columns]
            extra1: Sequence[str] = [n for n in df.columns if n not in self.field_names]
            df = df.loc[:, [*in_spec, *extra1]]
        if validate:
            rows = Rows.make(df, self.row_spec)
            for i in df.index:
                row = rows[i]
                rows.schema.validate(row)
        return df


@dataclass(frozen=True)
class Rows(Generic[_RowSpec]):
    df: pd.DataFrame
    row_spec: Type[_RowSpec]
    schema: Schema[_RowSpec]

    @staticmethod
    def make(df: pd.DataFrame, row_spec: Type[_RowSpec]) -> Rows[_RowSpec]:
        return Rows(df, row_spec, Schema.make(row_spec))

    @overload
    def __getitem__(self, index: int) -> _RowSpec:
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
            row = self.df.iloc[index.__index__()]
            content = {name: row[name] for name in self.schema.dtypes.keys()}
            return self.schema.row_spec(**content)
        elif isinstance(index, Iterable):
            return [self[i] for i in index]
        else:
            raise ValueError("Argument must be either an index or a sequence of indices")


def read_csv(
    filepath_or_buffer: Union[TextIO, str, bytes, os.PathLike],
    row_spec: Type[_RowSpec],
    order_columns: bool = True,
    missing_columns: Union[Literal["error"], Literal["missing"], Literal["fill"]] = "error",
    extra_columns: Union[Literal["drop"], Literal["keep"], Literal["error"]] = "drop",
    validate: bool = True,
    **kw_args,
) -> pd.DataFrame:
    """
    Reads a typed pandas DataFrame from a CSV file

    By "typed", we mean that a description of the columns of the dataframe is available, written
    either using a :class:`~typing.TypedDict` or a :mod:`dataclass`.

    Args:
        filepath_or_buffer: Path or open file to read from
        row_spec: Type specification, given either
        order_columns: Whether to order columns in the DataFrame as in the specification
        missing_columns: What to do with missing columns in the CSV file:

            - "error": raise an error (default)
            - "missing": leave the missing columns missing (set ``validate`` to False then)
            - "fill": fill the columns with the dtype default value

        extra_columns: What to do with extra columns present in the CSV file:

            - "drop": remove the extra columns from the dataframe (default)
            - "keep": keep the extra columns in the dataframe (note that the dtype is autodetected)
            - "error": raise an error

        validate: Whether to run validation on every row
                  (uses beartype if the `beartype <https://github.com/beartype/beartype>`_
                  otherwise runs simple :func:`isinstance` checks)
        kw_args: Additional keyword arguments not listed above are passed to :func:`pandas.read_csv`

    Returns:
        A pandas dataframe
    """
    schema = Schema.make(row_spec)
    df = pd.read_csv(
        filepath_or_buffer,
        dtype=schema.dtypes,
        **kw_args,
    )
    df = schema.process_raw_data_frame(
        df,
        missing_columns=missing_columns,
        extra_columns=extra_columns,
        validate=validate,
        order_columns=order_columns,
    )
    return df
