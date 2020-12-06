# Covasim docs

## Tutorials

Please see the `tutorials` subfolder.

## Everything else

This folder includes source code for building the docs. Users are unlikely to need to do this themselves. Instead, view the Covasim docs at http://docs.covasim.org.

To build the docs, follow these steps:

1.  Make sure dependencies are installed::
    ```
    pip install -r requirements.txt
    ```

2.  Make the documents; there are many build options. In most cases, running `./build_docs` (to rerun the tutorials; takes 2 min) or `./build_docs never` (does not rebuild the tutorials; takes 15 s) is best. Alternatively, one can call `make html` directly.

3.  The built documents will be in `./_build/html`.