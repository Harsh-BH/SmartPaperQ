from sympy.parsing.latex import parse_latex
from sympy import simplify
from sympy import pretty

def simplify_equation(latex_str: str) -> str:
    """
    Parse a LaTeX equation, simplify it, and return a human-readable string.
    """
    try:
        expr = parse_latex(latex_str)
        simp_expr = simplify(expr)
        return pretty(simp_expr)
    except Exception as e:
        return f"Error parsing/simplifying equation: {e}"
