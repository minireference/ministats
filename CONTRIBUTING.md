# Contributing

Contributions are welcome.

## Report bugs

Report bugs at <https://github.com/minireference/ministats/issues>.


## Implement features

Look through the GitHub issues for feature requests. 


### Write docs

We could always use more 
- Did the `README.md` make sense? What do you need to know at first glance?
- Does the `docs/` folder m
- 
Mini Statistics could always use more documentation, whether as part of
the official Mini Statistics docs, in docstrings, or even on the web in
blog posts, articles, and such.

## Submit Feedback

The best way to send feedback is to file an issue at
<https://github.com/minireference/ministats/issues>.





## Code contributions

This is how to set up `ministats` for local development.

1.  Fork the [ministats]{.title-ref} repo on GitHub.


2.  Clone your fork locally:

    ```
    $ git clone git@github.com:your-name/ministats.git
    ```

3.  Install your local copy into a virtualenv.
    Assuming you have virtualenvwrapper installed, this is how you set up your fork for
    local development:

    ```
    $ mkvirtualenv ministats
    $ cd ministats/
    $ python setup.py develop
    ```

1.  Create a branch for local development:

    ```
    $ git checkout -b name-of-your-bugfix-or-feature
    ```

    Now you can make your changes locally.

2.  When you\'re done making changes, check that your changes pass
    flake8 and the tests, including testing other Python versions with
    tox:

    ```
    $ flake8 ministats tests
    $ python setup.py test or pytest
    $ tox
    ```

    To get flake8 and tox, just pip install them into your virtualenv.

3.  Commit your changes and push your branch to GitHub:

    ```
    $ git add .
    $ git commit -m "Your detailed description of your changes."
    $ git push origin name-of-your-bugfix-or-feature
    ```

4.  Submit a pull request through the GitHub website.









## Pull Request Guidelines

Before you submit a pull request, check that it meets these guidelines:

1.  The pull request should include tests.
2.  If the pull request adds functionality, the docs should be updated.
    Put your new functionality into a function with a docstring, and add
    the feature to the list in README.rst.
3.  The pull request should work for Python 3.5, 3.6, 3.7 and 3.8, and
    for PyPy. Check
    <https://travis-ci.com/minireference/ministats/pull_requests> and
    make sure that the tests pass for all supported Python versions.


