from collections.abc import Sequence
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


# PLOT HELPERS
################################################################################

def plot_func(f, xlim=[0,5], flabel="f", ax=None):
    """
    Plot the function `f` over the interval `xlim`.
    """
    xs = np.linspace(xlim[0], xlim[1], 10000)
    fxs = fxs = np.array([f(x) for x in xs])
    ax = sns.lineplot(x=xs, y=fxs, ax=ax)    
    ax.set_xlim(*xlim)
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${flabel}(x)$")
    return ax


def plot_seq(ak, start=0, stop=10, label="$a_k$", ax=None):
    """
    Plot the sequence `ak` for between `start` and `stop`.
    """
    if ax is None:
        _, ax = plt.subplots()
    ks = np.arange(start, stop+1)
    aks = [ak(k) for k in ks]
    ax.stem(ks, aks, basefmt=" ")
    ax.set_xticks(ks)
    ax.set_xlabel("$k$")
    ax.set_ylabel(label)
    return ax





# CALCULUS OPERATIONS
################################################################################

def differentiate(f, x, delta=1e-9):
    """
    Compute the derivative of the function `f` at `x`
    using a slope calculation for very short `delta`.
    """
    df = f(x+delta) - f(x)
    dx = (x + delta) - x
    return df / dx


def integrate(f, a, b, n=10000):
    """
    Compute the area under the graph of `f`
    between `x=a` and `x=b` using `n` rectangles.
    """
    dx = (b - a) / n                       # width of rectangular strips
    xs = [a + k*dx for k in range(1,n+1)]  # right-corners of the strips
    fxs = [f(x) for x in xs]               # heights of the strips
    area = sum([fx*dx for fx in fxs])      # total area
    return area



# CALCULUS PLOT HELPERS
################################################################################


def plot_limit(f, xlim=[0,5], eps=0.00001, ylim=None, ax=None):
    """
    Plot the graph of the function `f` over the range(s) of values `xlim`.
    """
    # normalize `xlim` to be a list of x-ranges [[start,stop], ...]
    if isinstance(xlim[0], Sequence):
        xranages = xlim
    elif isinstance(xlim, Sequence) and len(xlim) == 2:
        xranages = [xlim]
    else:
        raise ValueError("Expected xlim to be limits or list of limits.")
    xmin, xmax = np.inf, -np.inf   # x-limits for the overall plot
    for xranage in xranages:
        xstart, xstop = xranage
        xs = np.linspace(xstart + eps, xstop - eps, 1000)
        fxs = np.array([f(x) for x in xs])
        ax = sns.lineplot(x=xs, y=fxs, ax=ax, color="C0")
        # record smallest and largest
        xmin = xstart if xstart <= xmin else xmin
        xmax = xstop if xstop > xmax else xmax
    ax.set_xlim(xmin, xmax)
    if ylim:
        ax.set_ylim(*ylim)
    return ax


def plot_ellipse(ax, x, y, width, height, c="C4", lw=0.6,
                 label=None, lx=0, ly=0, ha="left", va="center"):
    """
    Add to `ax` an ellipse centered at `(x,y)` of size `(width,height)`,
    and the optional text `label` at `(lx,ly)` with `ha`, `va` alignments.
    """
    ellipse = Ellipse((x, y), width=width, height=height,
                      zorder=10, facecolor='none',
                      edgecolor=c, linewidth=lw)
    ax.add_patch(ellipse)
    if label:
        ax.text(lx, ly, label, ha=ha, va=va)
    return ax



def plot_slope(f, x, delta=0, xlim=[0,5], ylim=None, ax=None):
    """
    Plot the graph of the function `f` over the interval `xlim`.
    Also plot the slope of function at `x` based on "run" `delta`.
    When `delta = 0`, the plot will show the derivative (instantaneous slope)
    When `delta != 0`, an approximate slope between f(x+delta)-f(x) / delta.
    """
    xs = np.linspace(*xlim, 1000)
    fxs = np.array([f(x) for x in xs])
    ax = sns.lineplot(x=xs, y=fxs, ax=ax, color="C0", label="$f(x)=x^2$")
    # Tangent line
    if delta == 0:
        dx = 1e-9
        dfdx = (f(x+dx) - f(x)) / dx
        T1xs = f(x) + dfdx*(xs - x)
        ax = sns.lineplot(x=xs, y=T1xs, ax=ax, color="red", linewidth=0.6)
        # point at x
        ax.plot(x, f(x), marker='o', markersize=2, color='red')
        dfdxstr = f"{dfdx:.3f}".rstrip("0").rstrip(".")
        halign = "right" if dfdx > 0 else "left"
        ax.text(x, f(x)+0.2, f"$f'({x})={dfdxstr}$", ha=halign, va="bottom", fontsize="small")
    # Average slope line
    else:
        y0 = f(x)
        y1 = f(x + delta)
        m = (y1 - y0) / delta
        b = f(x) - m*x
        yxs = m*xs + b
        mstr = f"{m:.3f}".rstrip("0").rstrip(".")
        bstr = f"{b:.3f}".rstrip("0").rstrip(".")
        sns.lineplot(x=xs, y=yxs, ax=ax, color="red", linewidth=0.6, label=f"$y = {mstr}x{bstr}$")
        # point at x
        ax.plot(x, f(x), marker='o', markersize=2, color='red')
        ax.text(x-0.1, f(x)-0.2, f"$({x},f({x}))$",
                ha="right", va="bottom", fontsize="small")
        # point at x+delta
        ax.plot(x+delta, f(x+delta), marker='o', markersize=2, color='red')
        x_plus_delta = f"{(x+delta):.3f}".rstrip("0").rstrip(".")
        ax.text(x+delta+0.1, f(x+delta)+0.3, f"$({x_plus_delta},f({x_plus_delta}))$",
                ha="right", va="bottom", fontsize="small")
        # run = delta
        fontsize = "small" if delta > 0.5 else "x-small"
        ax.plot([x, x + delta], [y0, y0], color="black", linewidth=0.5)
        ax.text(x + delta/2, y0 - 0.2, f"$\\Delta x$={delta}", ha="center", va="top", fontsize=fontsize)
        # rise = f(x+delta) - f(x)
        ax.plot([x + delta, x + delta], [y0, y1], color="black", linewidth=0.5)
        risestr = f"{(y1-y0):.3f}".rstrip("0").rstrip(".")
        ax.text(x + delta + 0.05, (y0 + y1)/2, f"$\\Delta y$={risestr}", ha="left", va="center", fontsize=fontsize)
    ax.set_xlim(*xlim)
    if ylim:
        ax.set_ylim(*ylim)
    return ax



def plot_integral(f, a=1, b=2, xlim=[0,5], flabel="f", ax=None, autolabel=False):
    """
    Plot the integral of `f` between `x=a` and `x=b`.
    """
    # Plot the function
    xs = np.linspace(xlim[0], xlim[1], 10000)
    fxs = fxs = np.array([f(x) for x in xs])
    ax = sns.lineplot(x=xs, y=fxs, ax=ax)    
    ax.set_xlim(*xlim)
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${flabel}(x)$")
    # Highlight the area under f(x) between x=a and x=b
    mask = (xs > a) & (xs < b)
    ax.fill_between(xs[mask], y1=fxs[mask], alpha=0.4)
    ax.vlines([a], ymin=0, ymax=f(a))
    ax.vlines([b], ymin=0, ymax=f(b))
    if autolabel:
        Alabel = f"$A_{{{flabel}}}({a},\\!{b})$"
        ax.text((a+b)/2, 0.4*f((a+b)/2), Alabel, ha="center", fontsize="large");
    return ax


def plot_riemann_sum(f, a=1, b=2, xlim=[0,5], n=20, flabel="f", ax=None):
    """
    Draw the Riemann sum approximation to the integral of `f`
    between `x=a` and `x=b` using `n` rectangles.
    """
    # Calculate the value of the Riemann sum approximation
    dx = (b - a) / n                       # width of rectangular strips
    xs = [a + k*dx for k in range(1,n+1)]  # right-corners of the strips
    fxs = [f(x) for x in xs]               # heights of the strips
    area = sum([fx*dx for fx in fxs])      # total area
    print(f"Riemann sum with n={n} rectangles: approx. area ≈ {area:.5f}")
    # Plot the function
    xs_plot = np.linspace(xlim[0], xlim[1], 10000)
    fxs_plot = f(xs_plot)
    ax = sns.lineplot(x=xs_plot, y=fxs_plot, ax=ax)
    # Draw rectangles
    left_corners = [xr - dx for xr in xs]
    ax.bar(left_corners, fxs, width=dx, align="edge", edgecolor="black", alpha=0.3)
    ax.set_xlim(*xlim)
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${flabel}(x)$")    
    return ax


def plot_series(ak, start=0, stop=10, label="$a_k$", ax=None):
    """
    Draw a bar plot that corresponds to the series `sum(ak)`
    between `start` and `stop`.
    """
    if ax is None:
        _, ax = plt.subplots()
    # Plot the sequence
    ks = np.arange(start, stop+1)
    aks = [ak(k) for k in ks]
    ax.stem(ks, aks, basefmt=" ")
    # Compute the sum
    area = sum(aks)
    print(f"The sum of the first {stop-start+1} terms of the sequence is {area:.6f}")
    # Draw the series as rectangles
    ax.bar(ks, aks, width=1, align="edge", edgecolor="black", alpha=0.3)
    ax.set_xticks(ks)
    ax.set_xlabel("$k$")
    ax.set_ylabel(label)
    return ax
