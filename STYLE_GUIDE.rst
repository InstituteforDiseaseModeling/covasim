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


2.21 Type Annotated Code (`GSG <https://google.github.io/styleguide/pyguide.html#221-type-annotated-code>`_)
----------------------------------------------------

**Difference**: Do **not** use type annotations.

**Reason**: Type annotations are useful for ensuring that simple functions do exactly what they're supposed to as part of a complex whole. They prioritize consistency over convenience, which is the correct priority for low-level library functions, but not for functions and classes that aim to make it as easy as possible for the user. 

For example, in Covasim, dates can be specified as numbers (``22``), strings (``'2022-02-02'``), or date objects (``datetime.date(2022, 2, 2)``), etc. Likewise, many quantities can be specified as a scalar, list, or array. If a function *usually* only needs a single input but can optionally operate on more than one, it adds burden on the user to require them to provide e.g. ``np.array([0.3])`` rather than just ``0.3``. In addition, most functions have default arguments of ``None``, in which case Covasim will use "sensible" defaults.

Attempting to apply type annotations to the flexibility Covasim gives to the user would result in monstrosities like ``start_day: typing.Union[None, str, int, dt.date, dt.datetime]``.



3.2 Line length (`GSG <https://google.github.io/styleguide/pyguide.html#32-line-length>`_)
----------------------------------------------------

**Difference**: Long lines are not *great*, but are justified in some circumstances.

**Reason**: Line lengths of 80 characters are due to `historical limitations <https://en.wikipedia.org/wiki/Characters_per_line>`_. Think of lines >80 characters as bad, but breaking a line as being equally bad. Decide whether a long line would be better implemented some other way -- for example, rather than breaking a >80 character list comprehension over multiple lines, use a ``for`` loop instead. Always keep literal strings together (do not use implicit string concatenation).

Line comments are encouraged in Covasim, and these can be as long as needed; they should not be broken over multiple lines to avoid breaking the flow of the code. A 50-character line with a 150 character line comment after it is completely fine.

Examples:

.. code-block:: python

    Yes: foo_bar(self, width, height, color='black', design=None, x='foo',
                 emphasis=None, highlight=0)

         if (width == 0 and height == 0 and
             color == 'red' and emphasis == 'strong'):



2.21 Type Annotated Code (`GSG <https://google.github.io/styleguide/pyguide.html#221-type-annotated-code>`_)
----------------------------------------------------

**Difference**: 

**Reason**: 