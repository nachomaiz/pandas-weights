# From pandas.core.accessor, modified so it doesn't cache the accessor

import warnings


class _Accessor:
    """
    Custom property-like object.

    A descriptor for caching accessors.

    Parameters
    ----------
    name : str
        Namespace that will be accessed under, e.g. ``df.foo``.
    accessor : cls
        Class with the extension methods.

    Notes
    -----
    For accessor, The class's __init__ method assumes that one of
    ``Series``, ``DataFrame`` or ``Index`` as the
    single argument ``data``.
    """

    def __init__(self, name: str, accessor) -> None:
        self._name = name
        self._accessor = accessor

    def __get__(self, obj, cls):
        # Commenting out caching behavior

        # if obj is None:
        #     # we're accessing the attribute of the class, i.e., Dataset.geo
        #     return self._accessor
        accessor_obj = self._accessor(obj)
        # Replace the property with the accessor object. Inspired by:
        # https://www.pydanny.com/cached-property.html
        # We need to use object.__setattr__ because we overwrite __setattr__ on
        # NDFrame
        # object.__setattr__(obj, self._name, accessor_obj)
        return accessor_obj


def register_accessor(name: str, cls):
    """
    Register a custom accessor on {klass} objects.

    Parameters
    ----------
    name : str
        Name under which the accessor should be registered. A warning is issued
        if this name conflicts with a preexisting attribute.

    Returns
    -------
    callable
        A class decorator.

    See Also
    --------
    register_dataframe_accessor : Register a custom accessor on DataFrame objects.
    register_series_accessor : Register a custom accessor on Series objects.
    register_index_accessor : Register a custom accessor on Index objects.

    Notes
    -----
    When accessed, your accessor will be initialized with the pandas object
    the user is interacting with. So the signature must be

    .. code-block:: python

        def __init__(self, pandas_object):  # noqa: E999
            ...

    For consistency with pandas methods, you should raise an ``AttributeError``
    if the data passed to your accessor has an incorrect dtype.

    >>> pd.Series(['a', 'b']).dt
    Traceback (most recent call last):
    ...
    AttributeError: Can only use .dt accessor with datetimelike values

    Examples
    --------
    In your library code::

        import pandas as pd

        @pd.api.extensions.register_dataframe_accessor("geo")
        class GeoAccessor:
            def __init__(self, pandas_obj):
                self._obj = pandas_obj

            @property
            def center(self):
                # return the geographic center point of this DataFrame
                lat = self._obj.latitude
                lon = self._obj.longitude
                return (float(lon.mean()), float(lat.mean()))

            def plot(self):
                # plot this array's data on a map, e.g., using Cartopy
                pass

    Back in an interactive IPython session:

        .. code-block:: ipython

            In [1]: ds = pd.DataFrame({{"longitude": np.linspace(0, 10),
               ...:                    "latitude": np.linspace(0, 20)}})
            In [2]: ds.geo.center
            Out[2]: (5.0, 10.0)
            In [3]: ds.geo.plot()  # plots data on a map
    """

    def decorator(accessor):
        if hasattr(cls, name):  # pragma: no cover
            warnings.warn(
                f"registration of accessor {repr(accessor)} under name "
                f"{repr(name)} for type {repr(cls)} is overriding a preexisting "
                f"attribute with the same name.",
                UserWarning,
                stacklevel=3,
            )
        setattr(cls, name, _Accessor(name, accessor))
        cls._accessors.add(name)
        return accessor

    return decorator
