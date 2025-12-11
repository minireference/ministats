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
    using the rise-over-run calculation for run `delta`.
    """
    df = f(x+delta) - f(x)
    dx = delta
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
