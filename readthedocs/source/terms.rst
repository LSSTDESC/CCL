*********************************************
Terms of Reference and Development Principles
*********************************************

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

------------------------------------------------------------------
Standards to establish the quality of code that can be part of CCL
------------------------------------------------------------------

- Contributions to CCL should be well-documented in python (only doc-string level docs are required, readthedocs does the rest). C code is expected to be commented, though not documented.

- Features added to CCL should pass required tests.

	1. one (publicly available) benchmark with an independent prediction should be provided to test the specific feature that has been added. The benchmark code can use CCL input as long as that is not affecting the validation procedure.

	2. Tests should run in continuous integration scheme used for CCL.

	3. Unit tests should be written following the suggestions of the PR reviewers.

- All PRs are code reviewed by at least one person (two are suggested for PRs that bring in new features or propose significant changes to the CCL API).

- CCL uses semantic versioning.

- We should aim to tag new versions with bug fixes on a monthly basis with the expectation that no release is made if no new bugs have been fixed.

- Critical bug fixes should be released almost immediately via bump of the micro version and a new release.

-------------------------------
Guidelines towards contribution
-------------------------------

- Contributions should be made via PRs, and such PRs should be reviewed as described above.

- Response to code review requests should be prompt and happen within a few days of being requested on github by the author of the PR. Code review by members of the CCL development team should happen within a week of the request being accepted.

- The CCL :ref:`devguide` should be consulted by those making PRs.

- Developers should use standard DESC channels to interact with the team (e.g., Slack, telecons, etc.).

- Developers should seek consensus with the team as needed about new features being incorporated (especially for API changes or additions). This process should happen via the CCL telecons, slack channel, or github comments generally before a new feature is merged. More minor changes, like bug fixes or refactoring, do not necessarily need this process.

- External contributions. They come in different forms: bug fixes or new features. They should proceed in full awareness of the LSST DESC Publication Policy. In particular, the use of new features added to the library, but not officially released, may require more formal steps to be taken within the DESC. External contributors and DESC members wishing to use CCL for non-DESC projects should consult with the TJP working group conveners, ideally before the work has started, but definitely before any publication or posting of the work to the arXiv.
