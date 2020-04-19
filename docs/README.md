# Covasim docs

This folder includes code for building the docs. Users are unlikely to need to
do this themselves.

To build the docs, follow these steps:

1.  Make sure dependencies are installed::
    ```
    pip install -r requirements.txt
    ```

2.  Make the documents; there are many build options, but most convenient is::

    ```
    make html
    ```

3.  The built documents will be in `./_build/html`.