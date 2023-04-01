""" Namespaces for constants and paths. """

import os


class UC:
    """ Namespace with a few UniCode characters for Greek symbols and ASCII art in stdout. """
    e = '\u03b5'  # epsilon
    t = '\u03c4'  # tau
    z = '\u03b6'  # zeta
    g = '\u03b3'  # gamma
    a = '\u03b1'  # alpha
    tl = '\u250c'  if os.platform == 'darwin' else '|'  # ┌
    bl = '\u2514'  if os.platform == 'darwin' else '|'  # └
    tr = '\u2510'  if os.platform == 'darwin' else '|'  # ┐
    br = '\u2518'  if os.platform == 'darwin' else '|'  # ┘
    hd = '\u2500'  if os.platform == 'darwin' else '-'  # ─
    vd = '\u2502'  if os.platform == 'darwin' else '|'  # │
    block = '\u25a0'  if os.platform == 'darwin' else '#'  # ■
    empty = '\u25a1'  if os.platform == 'darwin' else ' '  # □

class LC:
    """ Namespace with a few LaTeX commands for Greek symbols in plots. """
    e = r'$\varepsilon$'  # epsilon
    t = r'$\tau$'         # tau
    z = r'$\zeta$'        # zeta
    g = r'$\gamma$'       # gamma
    a = r'$\alpha$'       # alpha

class P:
    """ Namespace with all paths used in the project. """
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')) + os.sep
    plots = root + 'plots' + os.sep
    data = root + 'data' + os.sep
