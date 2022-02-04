===========
Style guide 
===========

In general, Covasim follows Google's `style guide <https://google.github.io/styleguide/pyguide.html>`_. If you simply follow that, you can't go too wrong. However, there are a few "house style" differences, which are described here.

Covasim uses ``pylint`` to ensure style conventions. To check if your styles are compliant, run ``./tests/check_style``.


Design philosophy
=================

Covasim's overall design philosophy is "**Common tasks should be simple**, while uncommon tasks can't always be simple, but still should be possible".

The audience for Covasim is *scientists*, not software developers. Assume that the average Covasim user dislikes coding and wants something that *just works*. Implications of this include:

- Commands should be short, simple, and obvious, e.g. ``cv.Sim().run().plot()``.
- If there's a "sensible" default value for something, use it. Users shouldn't *have* to think about false positivity rate or influenza-like illness prevalence if they just want to quickly add testing to their simulation via ``cv.test_prob(0.1)``.
- Be as flexible as possible with user inputs. If a user could only mean one thing, do that. If the user provides ``[0, 7, 14]`` but the function needs an array instead of a list, convert the list to an array automatically (``sc.toarray()`` exists for exactly this reason).
- Ensure the logic, especially the scientific logic, of the code is as clear as possible. If something is bad coding style but good science, you should probably do it anyway. 

The total work your code creates is:

.. math::

    W = \sum_p \left( u_p + n_p \times r_p + m_p \times e_p \right)

where:

- *W* is the total work
- *p* is each person
- *u* is the ramp-up time
- *n* is the number of reads
- *r* is the time per read
- *m* is the number of edits
- *e* is the time per edit

Common mistakes are to overemphasize *p* = 0 (you) over *p* > 0, and *e* (edit time) over *u* (ramp-up time) and *r* (read time).


House style
===========

As noted above, Covasim follows Google's style guide (GSG), with these exceptions (numbers refer to Google's style guide):


2.8 Default Iterators and Operators (`GSG <https://google.github.io/styleguide/pyguide.html#28-default-iterators-and-operators>`_)
----------------------------------------------------------------------------------------------------------------------------------

**Difference**: It's fine to use ``for key in obj.keys(): ...`` instead of ``for key in obj: ...``.

**Reason**: In larger functions with complex data types, it's not immediately obvious what type an object is. While ``for key in obj: ...`` is fine, especially if it's clear that ``obj`` is a dict, ``for key in obj.keys(): ...`` is also acceptable if it improves clarity.


2.21 Type Annotated Code (`GSG <https://google.github.io/styleguide/pyguide.html#221-type-annotated-code>`_)
------------------------------------------------------------------------------------------------------------

**Difference**: Do *not* use type annotations.

**Reason**: Type annotations are useful for ensuring that simple functions do exactly what they're supposed to as part of a complex whole. They prioritize consistency over convenience, which is the correct priority for low-level library functions, but not for functions and classes that aim to make it as easy as possible for the user. 

For example, in Covasim, dates can be specified as numbers (``22``), strings (``'2022-02-02'``), or date objects (``datetime.date(2022, 2, 2)``), etc. Likewise, many quantities can be specified as a scalar, list, or array. If a function *usually* only needs a single input but can optionally operate on more than one, it adds burden on the user to require them to provide e.g. ``np.array([0.3])`` rather than just ``0.3``. In addition, most functions have default arguments of ``None``, in which case Covasim will use "sensible" defaults.

Attempting to apply type annotations to the flexibility Covasim gives to the user would result in monstrosities like ``start_day: typing.Union[None, str, int, dt.date, dt.datetime]``.



3.2 Line length (`GSG <https://google.github.io/styleguide/pyguide.html#32-line-length>`_)
------------------------------------------------------------------------------------------

**Difference**: Long lines are not *great*, but are justified in some circumstances.

**Reason**: Line lengths of 80 characters are due to `historical limitations <https://en.wikipedia.org/wiki/Characters_per_line>`_. Think of lines >80 characters as bad, but breaking a line as being equally bad. Decide whether a long line would be better implemented some other way -- for example, rather than breaking a >80 character list comprehension over multiple lines, use a ``for`` loop instead. Always keep literal strings together (do not use implicit string concatenation).

Line comments are encouraged in Covasim, and these can be as long as needed; they should not be broken over multiple lines to avoid breaking the flow of the code. A 50-character line with a 150 character line comment after it is completely fine.

Examples:

.. code-block:: python

    # Yes: it's a bit longer than 80 chars but not too bad
    foo_bar(self, width, height, color='black', design=None, x='foo', emphasis=None)

    # No: the cost of breaking the line is too high
    foo_bar(self, width, height, color='black', design=None, x='foo',
            emphasis=None)

    # No: line is needlessly long, rename variables to be more concise
    foo_bar(self, object_width, object_height, text_color='black', text_design=None, x='foo', text_emphasis=None)

    # Yes: if you do need to break a line, make the break at a semantically meaningful point
    foo_bar(self, width, height, design=None, x='foo', emphasis=None,
            fg_color='black', bg_color='white', frame_color='orange')

    # Yes: long line comments are ok
    foo_bar(self, width, height, color='black', design=None, x='foo') # Note the difference with bar_foo(), which does not perform the opposite operation


3.6 Whitespace (`GSG <https://google.github.io/styleguide/pyguide.html#36-whitespace>`_)
----------------------------------------------------------------------------------------

**Difference**: You *should* use spaces to vertically align tokens.

**Reason**: This convention, which is a type of `semantic indenting <https://gist.github.com/androidfred/66873faf9f0b76f595b5e3ea3537a97c>`_, can greatly increase readability of the code by drawing attention to the semantic similarities and differences between consecutive lines.

Consider how hard it is to debug this code:

.. code-block:: python

    # Perform updates
    self.init_flows()
    self.flows['new_infectious'] += self.check_infectious()
    self.flows['new_symptomatic'] += self.check_symptomatic()
    self.flows['new_severe'] += self.check_symptomatic()
    self.flows['new_critical'] += self.check_critical()
    self.flows['new_recoveries'] += self.check_recovery()

vs. this:

.. code-block:: python

    # Perform updates
    self.init_flows()
    self.flows['new_infectious']  += self.check_infectious()
    self.flows['new_symptomatic'] += self.check_symptomatic()
    self.flows['new_severe']      += self.check_symptomatic()
    self.flows['new_critical']    += self.check_critical()
    self.flows['new_recoveries']  += self.check_recovery()

In the second case, the typo (repeated ``check_symptomatic()``)  immediately jumps out, whereas in the first case, it requires careful scanning of each line.

Vertically aligned code blocks also make it easier to edit code using editors that allow multiline editing (e.g., `Sublime <https://www.sublimetext.com/>`_). However, use your judgement -- there are cases where it does more harm than good, especially if the block is small, or if egregious amounts of whitespace would need to be used to achieve alignment:

.. code-block:: python

    # Yes
    test_prob  = 0.1 # Per-day testing probability
    vax_prob   = 0.3 # Per-campaign vaccination probability
    trace_prob = 0.8 # Per-contact probability of being traced

    # Yes
    t = 0 # Start day
    omicron_vax_prob = dict(low=0.05, high=0.1) # Per-day probability of receiving Omicron vaccine

    # Hell no
    t                = 0                        # Start day
    omicron_vax_prob = dict(low=0.05, high=0.1) # Per-day probability of receiving Omicron vaccine


2.21 Type Annotated Code (`GSG <https://google.github.io/styleguide/pyguide.html#221-type-annotated-code>`_)
----------------------------------------------------

**Difference**: 

**Reason**: 