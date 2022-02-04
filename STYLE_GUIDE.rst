===========
Style guide 
===========

In general, Covasim follows Google's `style guide <https://google.github.io/styleguide/pyguide.html>`_. If you simply follow that, you can't go too wrong. However, there are a few "house style" differences, which are described here.

Covasim uses ``pylint`` to ensure style conventions. To check if your styles are compliant, run ``./tests/check_style``.



House style
===========

As noted above, Covasim follows Google's style guide (GSG), with these exceptions (numbers refer to Google's style guide):


2.8 Default Iterators and Operators (`GSG <https://google.github.io/styleguide/pyguide.html#28-default-iterators-and-operators>`_)
----------------------------------------------------

**Difference**: It's fine to use ``for key in obj.keys(): ...`` instead of ``for key in obj: ...``.

**Reason**: In larger functions with complex data types, it's not immediately obvious what type an object is. While ``for key in obj: ...`` is fine, especially if it's clear that ``obj`` is a dict, ``for key in obj.keys(): ...`` is also acceptable if it improves clarity.