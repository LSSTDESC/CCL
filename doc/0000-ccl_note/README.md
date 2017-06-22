
## 0000-ccl_note
# Core Cosmology Library: Precision Cosmological Predictions for LSST

*David Alonso, Nora Elisa Chisari, Elizabeth Krause, C. Danielle Leonard, Sukhdeep Singh, Antonio Villarreal, Michal Vrastil, Joe Zuntz,  and others from the TJP Working Group*

An overview of the core cosmology library, providing routines for cosmological predictions with validated numerical accuracy.


## Editing this Paper

Fork and/or clone the project repo, and then
edit the primary file, `main.tex`.
Please use the `figures` folder for your images.

## Building this Paper

GitHub is our primary distributor for LSST DESC Notes:
once the Note has been merged into the project repo's master branch, it will be visible as a *shared* (but not *published*) paper. The presentation of Notes will be improved later, as the LSST DESC Publication System evolves.

You can compile latex papers locally with
```
make  [apj|apjl|prd|prl|mnras]
```
`make` with no arguments compiles latex in LSST DESC Note style, and uses Alex Drlica-Wagner's `mkauthlist` program to compile the author list from the information stored in `authors.csv`. The Makefile includes the `pip install` command needed to obtain `mkauthlist`, but this does mean that you'll need to have `pip` installed and able to install python packages on your system.

## Known building issues

If you are not admin of your system, you might need to edit the following line of the Makefile, replacing:
```
pip install --upgrade mkauthlist
```
by
```
pip install --user --upgrade mkauthlist
```

In this case, make sure that the directory where `mkauthlist` is installed is placed at the beginning of the `$PYTHONPATH`.

Also, `pip` might be installed in your system under a different alias. If so, you will need to replace `pip` above by the correct alias.

## Updating the Styles and Templates

From time to time, the latex style files will be updated: to re-download the latest versions, do
```
make update
```
This will over-write your folder's copies - but that's OK, as they are not meant to be edited by you!
The template files (`main.*` etc) are also likely to be updated; to get fresh copies of these files, do
```
make templates
```
However, since you will have edited at least one of the templates in your folder, `make templates` creates a special `templates` folder for you to refer to. Finally, to get *new* style or template files that are added to the `start_paper` project, you'll need to first get the latest `Makefile`, and then `make update` and/or `make templates`. The command to obtain the latest `Makefile` is
```
make new
```
This will add the latest `Makefile` to your `templates` folder. If you want to over-write your existing `Makefile`, you can do
```
make upgrade
```

## Automatic PDF Sharing

Once this project is in a public GitHub repo, we'll be able to use the `.travis.yml` file in this folder to cause [travis-ci](http://travis-ci.org) to compile the CCL Note into a PDF in the base repo at GitHub every time we push a commit to the master branch. The paper should then appear as:

**https://github.com/LSSTDESC/CCL/tree/pdf/0000-ccl_note.pdf**

To enable this service, we need to follow these steps:

1. Turn on travis continuous integration, by [toggling your repo on your travis profile](https://travis-ci.org/profile). If we don't see your repo listed, we may not have permission to do this: in this case, [we'll need to contact an admin via the issues](https://github.com/LSSTDESC/CCL/issues/new?body=@LSSTDESC/admin).
2. Get a [GitHub "personal access token"](https://github.com/settings/tokens), choosing the "repo" option.
3. Set the `GITHUB_API_KEY` environment variable with the value of this token at the CCL repo's [travis settings page](https://travis-ci.org/LSSTDESC/CCL/settings).
4. Merge the contents of the `.travis.yml` file in this folder with the existing CCL `.travis.yml` file).
5. Edit the final `git push` command with a repo admin GitHub username.  
6. Commit and push to trigger the travis build. Note that the PDF will only be deployed if the master branch is updated.
