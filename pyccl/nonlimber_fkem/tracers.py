"""Utilities for handling tracer collections in FKEM non-Limber work."""

from __future__ import annotations

from pyccl import lib, check

__all__ = ["build_tracer_collections"]


def build_tracer_collections(clt1, clt2):
    """Builds a pair of tracer collections for FKEM calculations.

    This method constructs two tracer collections from the provided
    tracer objects. It initializes new tracer collection structures and
    populates them with the tracers from the input collections.

    Args:
        clt1:
            The first tracer collection.
        clt2:
            The second tracer collection.

    Returns:
        tuple: A tuple containing the two built tracer collections.
            - The first tracer collection.
            - The second tracer collection.

    Raises:
        ValueError:
            If either input collection is ``None`` or contains no tracers.
        TypeError:
            If either input does not expose a ``_trc`` attribute and thus
            is not a valid tracer collection.
        RuntimeError:
            If allocation of the underlying C tracer collections fails or if
            the CCL backend reports an error while adding tracers.
    """
    # Require non-None inputs
    if clt1 is None or clt2 is None:
        raise ValueError("FKEM: 'clt1' and 'clt2' must not be None.")

    # Require the internal tracer lists
    if not hasattr(clt1, "_trc") or not hasattr(clt2, "_trc"):
        raise TypeError(
            "FKEM: both inputs must be tracer collections exposing '_trc'."
        )

    # Make sure there is at least one tracer in each collection
    if len(clt1._trc) == 0 or len(clt2._trc) == 0:
        raise ValueError(
            "FKEM: tracer collections must contain at least one tracer each."
        )

    status = 0

    t1, status = lib.cl_tracer_collection_t_new(status)
    check(status)
    t2, status = lib.cl_tracer_collection_t_new(status)
    check(status)

    if t1 is None or t2 is None:
        raise RuntimeError("FKEM: failed to allocate tracer collections.")

    for t in clt1._trc:
        status = lib.add_cl_tracer_to_collection(t1, t, status)
        check(status)

    for t in clt2._trc:
        status = lib.add_cl_tracer_to_collection(t2, t, status)
        check(status)

    return t1, t2
