# ---------------------------------------------------------
# Math Function Solver - final version
#
# What it does:
#   The user types a function of x, then chooses an operation
#   (from a dropdown, or by describing it in words).
#   The app computes the answer with SymPy and can show steps.
#
# Run with:  streamlit run app.py
# ---------------------------------------------------------

import re

import numpy as np
import pandas as pd
import streamlit as st
from sympy import (
    symbols, solve, diff, integrate, limit, latex, lambdify, factor, expand,
    simplify, cancel, nsimplify, Poly, Symbol, Integral, Mul, Pow, Add,
    E, sin, cos, tan,
)
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

# "x" is the variable our functions use.
x = symbols("x")

# These settings let the user type "3x" instead of "3*x"
# and "x^2" instead of "x**2".
TRANSFORMS = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)

# The names of the operations. They are used in the dropdown AND in the
# keyword dictionary, so they are stored once here to avoid typos.
OP_EVAL = "Evaluate at a point"
OP_ROOTS = "Find roots (where f(x) = 0)"
OP_DERIV = "Derivative"
OP_INTEGRAL = "Integral"
OP_DEFINITE = "Definite integral (with bounds)"
OP_LIMIT = "Limit"
OP_MINMAX = "Min / max (turning points)"
OP_SIMPLIFY = "Simplify / factor / expand"
OP_PLOT = "Plot"

OPERATIONS = [OP_EVAL, OP_ROOTS, OP_DERIV, OP_INTEGRAL, OP_DEFINITE,
              OP_LIMIT, OP_MINMAX, OP_SIMPLIFY, OP_PLOT]

# Words the user might type, and the operation each one means.
# ORDER MATTERS: the first operation with a matching word wins.
# For example "definite integral" must be checked before "integral",
# and "integral" before "derivative" (because "antiderivative"
# contains "derivative").
KEYWORDS = {
    OP_DEFINITE: ["definite", "integral from", "integral between",
                  "area between", "bounds"],
    OP_INTEGRAL: ["integral", "integrate", "antiderivative", "area under"],
    OP_LIMIT: ["limit", "approaches", "tends to", "as x ->"],
    OP_MINMAX: ["maximum", "minimum", "max", "min", "peak", "valley",
                "turning point", "critical point", "extrema", "extremum",
                "vertex", "highest", "lowest"],
    OP_DERIV: ["derivative", "differentiate", "diff", "d/dx", "slope",
               "rate of change"],
    OP_PLOT: ["plot", "graph", "draw", "show me"],
    OP_ROOTS: ["root", "zero", "solve", "crosses", "x-axis", "x axis",
               "intercept"],
    OP_SIMPLIFY: ["simplify", "simpler", "factor", "expand"],
    OP_EVAL: ["evaluate", "value", "plug", "calculate", "compute",
              "what is f"],
}

EXAMPLES = [
    "(choose an example)",
    "x^2 + 3x - 4",
    "x^3 - 3x",
    "sin(x)",
    "x*sin(x)",
    "e^x",
    "1/x",
    "(x^2 - 1)/(x - 1)",
    "sin(x)/x",
]

DIRECTIONS = {
    "Both sides": "+-",
    "From the right": "+",
    "From the left": "-",
}


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def parse_function(text):
    """Turn the text the user typed into a math expression SymPy understands."""
    text = text.strip()
    # If the user typed "f(x) = x^2", keep only the part after the "=".
    if "=" in text:
        text = text.split("=", 1)[1]
    return parse_expr(text, local_dict={"x": x, "e": E}, transformations=TRANSFORMS)


def parse_limit_point(text):
    """Turn text like '2', '-1' or 'inf' into a SymPy value."""
    text = text.strip().lower().replace("infinity", "oo").replace("inf", "oo")
    return parse_expr(text, transformations=TRANSFORMS)


def word_in(word, request):
    """Check if a keyword appears in the request.
    Short words (like 'min' or 'root') must match as whole words,
    so that 'minus' does not count as 'min'."""
    if len(word) <= 4:
        return re.search(r"\b" + re.escape(word) + r"s?\b", request) is not None
    return word in request


def pick_operation(request):
    """Look through the user's sentence and return the matching operation.
    Returns None if no keyword is found."""
    request = request.lower()
    for operation_name, words in KEYWORDS.items():
        for word in words:
            if word_in(word, request):
                return operation_name
    return None


def find_numbers(text):
    """Find all numbers in a sentence, e.g. 'from -5 to 5' -> [-5.0, 5.0]."""
    return [float(n) for n in re.findall(r"-?\d+(?:\.\d+)?", text)]


def nice_number(value):
    """Show 3.0 as 3, but keep 2.5 as 2.5."""
    return int(value) if float(value).is_integer() else value


def to_sym(value):
    """Turn a number from a box into an exact SymPy number (0.5 -> 1/2)."""
    return nsimplify(nice_number(value), rational=True)


def step(text, formula=None):
    """Show one line of explanation, with an optional formula under it."""
    st.markdown(text)
    if formula:
        st.latex(formula)


def derivative_rule(term):
    """Name the main rule used to differentiate one term."""
    if not term.has(x):
        return "constant rule: the derivative of a constant is 0"
    if term == x:
        return "the derivative of x is 1"
    if isinstance(term, Mul):
        x_factors = [a for a in term.args if a.has(x)]
        if len(x_factors) == 1:
            return "constant multiple rule: keep the constant, differentiate the rest"
        if any(isinstance(a, Pow) and a.exp.is_negative for a in x_factors):
            return "quotient rule: (u/v)' = (u'v - uv') / v^2"
        return "product rule: (u*v)' = u'v + uv'"
    if isinstance(term, Pow):
        if term.base == x and term.exp.is_number:
            return "power rule: d/dx[x^n] = n * x^(n-1)"
        return "chain rule (together with the power rule)"
    if term.args and term.args[0] != x:
        return "chain rule: differentiate the outside, then multiply by the derivative of the inside"
    return "standard derivative of a known function"


def integral_rule(term):
    """Name the main rule used to integrate one term."""
    if not term.has(x):
        return "constant rule: the integral of a constant c is c*x"
    if term == x:
        return "power rule with n = 1"
    if isinstance(term, Mul) and len([a for a in term.args if a.has(x)]) == 1:
        return "constant multiple rule: keep the constant, integrate the rest"
    if isinstance(term, Pow) and term.base == x and term.exp.is_number:
        if term.exp == -1:
            return "special case: the integral of 1/x is ln|x|"
        return "power rule: the integral of x^n is x^(n+1) / (n+1)"
    return "standard integral (or a technique such as substitution or by parts)"


def classify_turning_point(second_derivative_value):
    """Second derivative test: what kind of turning point is it?"""
    value = simplify(second_derivative_value)
    if value.is_positive:
        return "Local minimum (second derivative is positive)"
    if value.is_negative:
        return "Local maximum (second derivative is negative)"
    return "Inconclusive (second derivative is 0 or unclear: check the graph)"


# ---------------------------------------------------------
# The page the user sees
# ---------------------------------------------------------
st.title("Math Function Solver")

with st.expander("How to use this app"):
    st.markdown(
        "1. Type a function of x (for example `x^2 + 3x - 4`), or pick an example.\n"
        "2. Choose what to do with it from the dropdown, **or** describe it in "
        'words (for example "find the slope" or "plot from -5 to 5").\n'
        "3. Turn on **Show detailed steps** if you want the explanation.\n"
        "4. Click **Solve**."
    )

# 1) A box where the user types the function (and an examples dropdown).
if "function_text" not in st.session_state:
    st.session_state["function_text"] = "x^2 + 3x - 4"


def load_example():
    """Runs when the user picks an example: copy it into the function box."""
    choice = st.session_state["example_choice"]
    if choice != EXAMPLES[0]:
        st.session_state["function_text"] = choice


text = st.text_input("Enter a function of x", key="function_text")
st.selectbox("Examples", EXAMPLES, key="example_choice", on_change=load_example)

# 2) A dropdown where the user chooses what to do with it.
operation = st.selectbox("What do you want to do?", OPERATIONS)

# 2b) Optional: the user can describe what they want in words instead.
#     If we understand it, it replaces the dropdown choice.
request = st.text_input(
    "Or describe what you want (optional)",
    placeholder='for example: "find the slope" or "plot from -5 to 5"',
)
numbers = []
if request.strip():
    match = pick_operation(request)
    numbers = find_numbers(request)
    if match:
        operation = match
        st.info("Understood: " + operation)
        if numbers:
            st.caption("Numbers found in your request: "
                       + ", ".join(str(nice_number(n)) for n in numbers))
    else:
        st.warning(
            "I didn't understand that request, so I'm using the dropdown choice. "
            'Try words like "derivative", "integral", "limit", "maximum", '
            '"roots", "plot" or "simplify".'
        )

# 3) Extra boxes for operations that need them.
#    If the user's sentence contained numbers, use them as the starting values.
point = None
if operation == OP_EVAL:
    default_point = numbers[0] if numbers else 0.0
    point = st.number_input("Value of x", value=float(default_point))

lower, upper = 0.0, 1.0
if operation == OP_DEFINITE:
    if len(numbers) >= 1:
        lower = numbers[0]
    if len(numbers) >= 2:
        upper = numbers[1]
    lower = st.number_input("Lower bound a", value=float(lower))
    upper = st.number_input("Upper bound b", value=float(upper))

limit_text, direction = "0", "Both sides"
if operation == OP_LIMIT:
    lowered = request.lower()
    if "infinity" in lowered or "inf" in lowered:
        limit_text = "-oo" if "-inf" in lowered or "negative inf" in lowered else "oo"
    elif numbers:
        limit_text = str(nice_number(numbers[0]))
    limit_text = st.text_input("x approaches (type a number, or 'inf' for infinity)",
                               value=limit_text)
    direction = st.selectbox("From which side?", list(DIRECTIONS.keys()))

x_min, x_max = -10.0, 10.0
if operation == OP_PLOT:
    if len(numbers) >= 2:
        x_min, x_max = numbers[0], numbers[1]
    x_min = st.number_input("Plot from x =", value=float(x_min))
    x_max = st.number_input("Plot to x =", value=float(x_max))

# 4) The toggle for detailed steps.
show_steps = st.toggle("Show detailed steps")

# 5) The button that runs everything.
#    We remember that Solve was clicked, so flipping the toggle
#    doesn't make the answer disappear.
if st.button("Solve"):
    st.session_state["solved"] = True

if st.session_state.get("solved"):
    try:
        f = parse_function(text)

        st.subheader("Your function")
        st.latex("f(x) = " + latex(f))

        st.subheader("Result")

        # ---------------- Evaluate ----------------
        if operation == OP_EVAL:
            result = f.subs(x, to_sym(point))
            st.latex(f"f({nice_number(point)}) = " + latex(result))
            try:
                st.write("Decimal value:", float(result))
            except Exception:
                pass

            if show_steps:
                st.subheader("Steps")
                # Put the number in place of x, without calculating yet.
                shown = str(nice_number(point))
                if point < 0:
                    shown = "(" + shown + ")"
                substituted = f.xreplace({x: Symbol(shown)})
                step("1. Replace every x with the value:",
                     f"f({nice_number(point)}) = " + latex(substituted))
                step("2. Calculate:", "= " + latex(result))

        # ---------------- Roots ----------------
        elif operation == OP_ROOTS:
            roots = solve(f, x)
            if len(roots) == 0:
                st.write("No roots found.")
            for r in roots:
                st.latex("x = " + latex(r))

            if show_steps:
                st.subheader("Steps")
                step("1. A root is a value of x that makes the function equal 0:",
                     latex(f) + " = 0")

                factored = factor(f)
                is_poly = f.is_polynomial(x)
                degree = Poly(f, x).degree() if is_poly else None

                if isinstance(factored, Mul) and factored != f:
                    step("2. Factor the function:", "f(x) = " + latex(factored))
                    step("3. A product is zero when one of its factors is zero, "
                         "so set each factor equal to 0:")
                    for part in Mul.make_args(factored):
                        if part.has(x):
                            base = part.base if isinstance(part, Pow) else part
                            sols = solve(base, x)
                            st.latex(latex(base) + r" = 0 \;\Rightarrow\; x = "
                                     + ", ".join(latex(s) for s in sols))
                elif is_poly and degree == 2:
                    a, b, c = Poly(f, x).all_coeffs()
                    disc = b**2 - 4 * a * c
                    step("2. This is a quadratic, ax^2 + bx + c. Read off the coefficients:",
                         rf"a = {latex(a)},\quad b = {latex(b)},\quad c = {latex(c)}")
                    step("3. Compute the discriminant:",
                         r"\Delta = b^2 - 4ac = " + latex(disc))
                    if disc.is_negative:
                        step("The discriminant is negative, so there are no real roots "
                             "(only complex ones).")
                    step("4. Use the quadratic formula:",
                         r"x = \frac{-b \pm \sqrt{\Delta}}{2a}")
                else:
                    step("2. This function can't be factored simply, so SymPy "
                         "solves f(x) = 0 directly.")

                if roots:
                    step("Solutions:", ",\\quad ".join("x = " + latex(r) for r in roots))

        # ---------------- Derivative ----------------
        elif operation == OP_DERIV:
            result = diff(f, x)
            st.latex(r"f'(x) = " + latex(result))

            if show_steps:
                st.subheader("Steps")
                terms = Add.make_args(f)
                step("1. Start with the function:", "f(x) = " + latex(f))
                if len(terms) > 1:
                    step("2. Sum rule: differentiate each term separately, then add the results.")
                else:
                    step("2. The function is a single term, so we differentiate it directly.")
                for t in terms:
                    step("- Term: rule used is the **" + derivative_rule(t) + "**",
                         r"\frac{d}{dx}\left[" + latex(t) + r"\right] = " + latex(diff(t, x)))
                step("3. Combine (and simplify):", r"f'(x) = " + latex(result))

        # ---------------- Integral ----------------
        elif operation == OP_INTEGRAL:
            result = integrate(f, x)
            st.latex(r"\int f(x)\,dx = " + latex(result) + " + C")

            if show_steps:
                st.subheader("Steps")
                terms = Add.make_args(f)
                step("1. Start with the function:", "f(x) = " + latex(f))
                if len(terms) > 1:
                    step("2. Sum rule: integrate each term separately, then add the results.")
                else:
                    step("2. The function is a single term, so we integrate it directly.")
                for t in terms:
                    step("- Term: rule used is the **" + integral_rule(t) + "**",
                         r"\int " + latex(t) + r"\,dx = " + latex(integrate(t, x)))
                step("3. Combine and add the constant of integration C:",
                     r"\int f(x)\,dx = " + latex(result) + " + C")

        # ---------------- Definite integral ----------------
        elif operation == OP_DEFINITE:
            a_sym, b_sym = to_sym(lower), to_sym(upper)
            result = integrate(f, (x, a_sym, b_sym))
            st.latex(latex(Integral(f, (x, a_sym, b_sym))) + " = " + latex(result))
            try:
                st.write("Decimal value:", float(result))
            except Exception:
                pass

            if show_steps:
                st.subheader("Steps")
                F = integrate(f, x)
                step("1. Find the antiderivative F(x) (the indefinite integral):",
                     "F(x) = " + latex(F))
                step("2. Evaluate F at the upper bound and the lower bound, "
                     "then subtract: F(b) - F(a).",
                     f"F({latex(b_sym)}) - F({latex(a_sym)}) = "
                     + latex(F.subs(x, b_sym)) + " - \\left(" + latex(F.subs(x, a_sym)) + "\\right)")
                step("3. Result:", "= " + latex(result))

        # ---------------- Limit ----------------
        elif operation == OP_LIMIT:
            try:
                a_val = parse_limit_point(limit_text)
            except Exception:
                a_val = None
                st.error("I couldn't understand that value. Try 0, 2, -1 or inf.")

            if a_val is not None:
                result = None
                try:
                    if direction == "Both sides" and a_val.is_finite:
                        # Check each side separately. If they disagree,
                        # the (two-sided) limit does not exist.
                        left = limit(f, x, a_val, dir="-")
                        right = limit(f, x, a_val, dir="+")
                        if left == right:
                            result = left
                        else:
                            st.write("The limit does not exist: the left and right "
                                     "sides give different results.")
                            st.latex(rf"\lim_{{x \to {latex(a_val)}^{{-}}}} f(x) = {latex(left)}"
                                     rf",\quad \lim_{{x \to {latex(a_val)}^{{+}}}} f(x) = {latex(right)}")
                    else:
                        result = limit(f, x, a_val, dir=DIRECTIONS[direction])

                    if result is not None:
                        st.latex(rf"\lim_{{x \to {latex(a_val)}}} f(x) = " + latex(result))
                        if direction != "Both sides":
                            st.caption("Taken " + direction.lower() + ".")
                except ValueError:
                    st.write("The limit does not exist: the left and right sides "
                             "give different results.")

                if show_steps and result is not None:
                    st.subheader("Steps")
                    direct = None
                    try:
                        direct = f.subs(x, a_val)
                    except Exception:
                        pass
                    if direct is not None and direct.is_finite:
                        step("1. Substitute the value directly into the function:",
                             rf"f({latex(a_val)}) = " + latex(direct))
                        step("2. It gives a finite number, so that is the limit.")
                    else:
                        step("1. Substituting the value directly gives an undefined or "
                             "infinite result (for example 0/0), so we need another method.")
                        simpler = cancel(f) if f.is_rational_function(x) else simplify(f)
                        if simpler != f:
                            step("2. Simplify the function first (for example cancel "
                                 "common factors):", "f(x) = " + latex(simpler))
                            step("3. Then take the limit of the simplified function.")
                        else:
                            step("2. Use limit techniques (for example L'Hopital's rule "
                                 "or comparing growth rates). SymPy does this automatically.")
                    step("Result:", rf"\lim_{{x \to {latex(a_val)}}} f(x) = " + latex(result))

        # ---------------- Min / max ----------------
        elif operation == OP_MINMAX:
            f1 = diff(f, x)
            f2 = diff(f1, x)
            critical = [s for s in solve(f1, x) if s.is_real]

            if not critical:
                st.write("No turning points found (the function has no point "
                         "where the slope is 0).")
            for s in critical:
                value = simplify(f.subs(x, s))
                st.latex(rf"x = {latex(s)},\quad f(x) = {latex(value)}")
                st.write(classify_turning_point(f2.subs(x, s)))
            if f.has(sin, cos, tan):
                st.caption("Periodic functions have infinitely many turning points; "
                           "only the ones SymPy finds are listed.")

            if show_steps:
                st.subheader("Steps")
                step("1. Turning points happen where the slope is zero. "
                     "Find the derivative:", r"f'(x) = " + latex(f1))
                step("2. Set the derivative equal to 0 and solve for x:",
                     latex(f1) + " = 0")
                if critical:
                    step("Critical points:",
                         ",\\quad ".join("x = " + latex(s) for s in critical))
                    step("3. Second derivative test: find f''(x). If it is positive "
                         "at a point it is a minimum, if negative it is a maximum.",
                         r"f''(x) = " + latex(f2))
                    for s in critical:
                        st.latex(rf"f''({latex(s)}) = " + latex(simplify(f2.subs(x, s))))
                else:
                    step("There are no real solutions, so there are no turning points.")

        # ---------------- Simplify / factor / expand ----------------
        elif operation == OP_SIMPLIFY:
            step("**Simplified:**", latex(simplify(f)))
            step("**Factored:**", latex(factor(f)))
            step("**Expanded:**", latex(expand(f)))

            if show_steps:
                st.subheader("Steps")
                step("- **Simplify** rewrites the expression in its simplest equivalent form "
                     "(for example cancelling common factors).")
                step("- **Factor** writes it as a product, like x^2 - 1 = (x - 1)(x + 1).")
                step("- **Expand** multiplies everything out, like (x + 1)^2 = x^2 + 2x + 1.")

        # ---------------- Plot ----------------
        elif operation == OP_PLOT:
            if x_min >= x_max:
                st.error("The start of the range must be smaller than the end.")
            else:
                # Make 500 x-values between x_min and x_max.
                xs = np.linspace(x_min, x_max, 500)
                # Turn the SymPy function into a fast numeric function.
                f_numeric = lambdify(x, f, "numpy")
                # np.full_like handles constant functions like "5".
                ys = np.full_like(xs, f_numeric(xs), dtype=float)
                # Replace "infinity" values (like 1/x at 0) with gaps in the line.
                ys[np.isinf(ys)] = np.nan
                # Draw it.
                chart_data = pd.DataFrame({"f(x)": ys}, index=xs)
                st.line_chart(chart_data)

                if show_steps:
                    st.subheader("Steps")
                    step("1. Choose the range of x values to draw: "
                         f"from {nice_number(x_min)} to {nice_number(x_max)}.")
                    step("2. Create 500 evenly spaced x values in that range.")
                    step("3. Calculate f(x) for each of those x values.")
                    step("4. Skip points where f(x) is undefined (for example 1/x at x = 0), "
                         "which leaves a gap in the line.")
                    step("5. Connect the points to draw the curve.")
                    try:
                        y0 = f.subs(x, 0)
                        if y0.is_real:
                            step("Extra: the graph crosses the y-axis at:",
                                 "f(0) = " + latex(y0))
                    except Exception:
                        pass

    except Exception:
        st.error(
            "Sorry, I couldn't understand that function. "
            "Try something like: x^2 + 3x - 4  or  sin(x) * x"
        )
