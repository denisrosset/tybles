Tybles: simple schemas for Pandas dataframes
============================================

Tybles is a simple layer over `Pandas <https://pandas.pydata.org/>`_ that:

- **documents** the dataframe schema using standard Python
  `dataclasses <https://docs.python.org/3/library/dataclasses.html>`_

- **sets up** Pandas parsing using numpy's :class:`numpy.dtype`

- **validates** the resulting dataframe using either 
  `beartype <https://github.com/beartype/beartype>`_ or
  `typeguard <https://github.com/agronholm/typeguard>`_.

To install
----------

Just use `pip`::

    pip install tybles

If you want to pull the `beartype <https://github.com/beartype/beartype>`_ or
  `typeguard <https://github.com/agronholm/typeguard>`_ extras, add either or both::

    pip install tybles[beartype, typeguard]

To use
------

Have a look at the :ref:`tutorial <tutorial>`.


.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: General information

    Home page <self>
    tutorial


.. toctree::
    :hidden:
    :maxdepth: 3
    :caption: API

    api
