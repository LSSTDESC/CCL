************
Scope of CCL
************

This document lays out terms of reference for CCL developers and contributors in the wider TJP and LSST DESC context.

----------
Background
----------

CCL was created to provide theoretical predictions of large-scale structure statistics for DESC cosmological analyses but it has also become a publicly available tool.

-----
Scope
-----

CCL has a commitment to support key DESC analyses. In this regard, CCL aims to

- support cosmological models that are priorities for WGs (e.g., as defined by the "Beyond wCDM" taskforce);

- ensure sufficient flexibility such that modifications can be implemented at a later stage as needed (e.g., models for observational systematics or changes to how the non-linear power spectrum is computed for various parts of the analysis);

- support core predictions (e.g., shear correlation functions), but not necessarily every statistic that could be computed from those core predictions (e.g., COSEBIs).

CCL interface support is as follows.

- Public APIs are maintained in Python only (i.e., version bumping is not necessary following changes in the C interface).

- The C implementation can be used for speed as needed.

- Documentation should be maintained in the main README, readthedocs and the benchmarks folder.

- Examples will be maintained in a separate repository.

Boundaries of CCL (with respect to various models and systematics)

- Not all models need to have equal support. CCL aims to support the DESC analyses, not to support all possible statistics in all models.

- CCL should provide a generic framework to enable modeling of systematics effects. However, overly specific models for systematics, for example, for galaxy bias, detector effects or intrinsic alignments, are best located in other repositories (e.g., firecrown).

Boundaries of CCL (with respect to other TJP software)

- TJP will necessarily develop other software tools to support DESC cosmological analyses. CCL development should ensure internal consistency between the various tools by defining a common parameterization of theoretical quantities (e.g., kernels for Limber integration, the metric potentials, etc.).

- CCL development should also proceed in a manner to avoid significant duplication of effort within the collaboration.
