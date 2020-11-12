'''
Define options for Covasim.
'''


import os

# Set default arithmetic precision -- use 32-bit by default for speed and memory efficiency
precision = int(os.getenv('COVASIM_PRECISION', 32))

# Set the default font size -- if 0, use Matplotlib default
font_size = int(os.getenv('COVASIM_FONT_SIZE', 0))