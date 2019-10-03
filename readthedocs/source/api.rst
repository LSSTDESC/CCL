*************************
API Changes and Stability
*************************

CCL follows the conventions of `semantic versioning <https://semver.org/>`_.
Semantic versioning is a consistent set of rules for constructing CCL version
numbers so that CCL users can understand when new releases have backwards
compatible APIs.

An API breakage is anything that modifies the name or call signature of an
existing public-facing function or variable in the ``Python`` code in such a way
that the change is not backwards compatible. For example, removing a ``Python`` keyword
argument is an API breakage. However, adding one is not. Significant changes to the
behavior of a function also count as an API breakage (e.g. if an assumption is
changed that causes different results to be produced by default). Fixing a bug
that affects the output of a function does not count as an API breakage, unless
the fix modifies function names and arguments too as described above.

If the API is broken, the CCL major version number must be incremented (e.g. from
v1.5.0 to v2.0.0), and a new tagged release should be made shortly after the
changes are pushed to master. The new release must include notes in
``CHANGELOG.md`` that describe how the API has been changed. The aim is to make
it clear to users how their CCL-based code might be affected if they update to
the latest version, especially if the update is likely to break something. All
API changes should be discussed with the CCL team before being pushed to the
``master`` branch.
