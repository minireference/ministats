from collections.abc import Sequence
from matplotlib.patches import Ellipse
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns



# PLOT HELPERS
################################################################################

def plot_func(f, xlim=[0,5], ylim=None, flabel="f", ax=None):
    """
    Plot the function `f` over the interval `xlim`.
    """
    xs = np.linspace(xlim[0], xlim[1], 10000)
    fxs = fxs = np.array([f(x) for x in xs])
    ax = sns.lineplot(x=xs, y=fxs, ax=ax)    
    ax.set_xlim(*xlim)
    ax.set_xlabel("$x$")
    ax.set_ylabel(f"${flabel}(x)$")
    if ylim:
        ax.set_ylim(*ylim)
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
    Compute the derivative of the function `f` at `x` using
    the slope calculation with a very short step `delta`.
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
                 label=None, lx=0, ly=0, ha="left", va="center", fontsize="small"):
    """
    Add to `ax` an ellipse centered at `(x,y)` of size `(width,height)`,
    and the optional text `label` at `(lx,ly)` with `ha`, `va` alignments.
    """
    ellipse = Ellipse((x, y), width=width, height=height,
                      zorder=10, facecolor='none',
                      edgecolor=c, linewidth=lw)
    ax.add_patch(ellipse)
    if label:
        ax.text(lx, ly, label, ha=ha, va=va, fontsize=fontsize)
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
        sns.lineplot(x=xs, y=yxs, ax=ax, color="red", linewidth=0.6, label=f"slope = {mstr}")
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



# MULTIVARIABLE CALCULUS FIGURES
################################################################################

def plot_slices_through_paraboloid(direction="x", xmax=2.01, ymax=4.02,
                                   ngrid=400, fig=None):
    """
    Plot slices through the surface z = 4 - x^2 - y^2/4.

    direction = "x": slices at fixed x, varying y.
        xs = [-1.9, -1.5, -1, -0.5, 0, 0.5, 1, 1.5, 1.9]
    direction = "y": slices at fixed y, varying x.
        ys = [-3.9, -3, -2, -1, 0, 1, 2, 3, 3.9]
    """
    from .book.figures import find_nearest1
    from .book.figures import polygon_under_graph
    from matplotlib.collections import PolyCollection
    if fig is None:
        fig = plt.figure(figsize=(7, 4))
    ax = fig.add_subplot(projection="3d")

    # Grid on x,y plane
    x = np.linspace(-xmax, xmax, ngrid)
    y = np.linspace(-ymax, ymax, ngrid)
    X, Y = np.meshgrid(x, y, indexing="xy")
    Z = 4 - X**2 - Y**2 / 4.0

    # Only keep the "cap" z >= 0
    Z = np.maximum(Z, 0.0)

    if direction == "x":
        cuts = np.array([-1.9, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 1.9])
        # Polygons live in (y,z)-plane; extruded along x
        verts = []
        for xcut in cuts:
            idx = find_nearest1(x, xcut)        # column in X,Z
            z_slice = Z[:, idx]                 # as function of y
            vert = polygon_under_graph(y, z_slice)
            verts.append(vert)
        facecolors = plt.colormaps["viridis_r"](np.linspace(0, 1, len(verts)))
        poly = PolyCollection(verts, facecolors=facecolors, alpha=0.6)
        ax.add_collection3d(poly, zs=cuts, zdir="x")

    elif direction == "y":
        cuts = np.array([-3.9, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, 3.9])
        # Polygons live in (x,z)-plane; extruded along y
        verts = []
        for ycut in cuts:
            idx = find_nearest1(y, ycut)        # row in Y,Z
            z_slice = Z[idx, :]                 # as function of x
            vert = polygon_under_graph(x, z_slice)
            verts.append(vert)
        facecolors = plt.colormaps["viridis_r"](np.linspace(0, 1, len(verts)))
        poly = PolyCollection(verts, facecolors=facecolors, alpha=0.6)
        ax.add_collection3d(poly, zs=cuts, zdir="y")

    else:
        raise ValueError("direction must be 'x' or 'y'")

    # Axes settings
    # ax.set_box_aspect((2 * xmax, 2 * ymax, 4))  # roughly match geometry
    ax.set(
        xlim=(-xmax, xmax),
        ylim=(-ymax, ymax),
        zlim=(0, 4.0),
        xlabel="$x$",
        ylabel="$y$",
        zlabel="$z$",
    )

    return ax


def plot_point_charge_field(elev=20, azim=40, grid_lim=1.5, n_points=6):
    """
    Plot an E vector field around a point charge at the origin.
    Custom logic to make the plot more readable:
    - compressed grid near origin
    - gentle magnitude scaling
    - view-dependent filtering of short arrows
    """
    from mpl_toolkits.mplot3d import proj3d
    from matplotlib.patches import FancyArrowPatch

    class Arrow3D(FancyArrowPatch):
        """
        A 3D arrow used to represent vectors in 3D.
        xs, ys, zs are length-2 lists: [start, end].
        """
        def __init__(self, xs, ys, zs, *args, **kwargs):
            super().__init__((0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs

        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            x2d, y2d, _ = proj3d.proj_transform(
                xs3d, ys3d, zs3d, self.axes.get_proj()
            )
            self.set_positions((x2d[0], y2d[0]), (x2d[1], y2d[1]))
            super().draw(renderer)

        def do_3d_projection(self, renderer=None):
            xs3d, ys3d, zs3d = self._verts3d
            return float(np.mean(zs3d))
        
    # ------------------------------------------------------------
    # compressed grid → innermost points closer to origin
    # ------------------------------------------------------------
    u = np.linspace(-1, 1, n_points)
    coords = grid_lim * np.sign(u) * (np.abs(u)**1.5)

    X, Y, Z = np.meshgrid(coords, coords, coords)

    # radial distance
    R = np.sqrt(X**2 + Y**2 + Z**2)
    R_safe = np.where(R < 1e-9, 1e-9, R)

    # unit direction
    Ex = X / R_safe
    Ey = Y / R_safe
    Ez = Z / R_safe

    # gentle scaling
    S = 1 / (1 + 2 * R_safe**2)
    Ex *= S
    Ey *= S
    Ez *= S

    # flatten for filtering
    x0 = X.flatten()
    y0 = Y.flatten()
    z0 = Z.flatten()
    dx = Ex.flatten()
    dy = Ey.flatten()
    dz = Ez.flatten()

    # arrow lengths
    L = np.sqrt(dx**2 + dy**2 + dz**2)

    # ------------------------------------------------------------
    # view-dependent filtering (drop short front/back arrows)
    # ------------------------------------------------------------
    phi = np.deg2rad(elev)
    theta = np.deg2rad(azim)

    # approximate view direction vector
    v = np.array([
        np.cos(phi) * np.cos(theta),
        np.cos(phi) * np.sin(theta),
        np.sin(phi),
    ])

    # position along view vector
    d = x0 * v[0] + y0 * v[1] + z0 * v[2]

    # thresholds
    L_thresh = np.quantile(L, 0.4)   # bottom 40% = short
    d_front  = np.quantile(d, 0.7)   # closest ~30%
    d_back   = np.quantile(d, 0.3)   # farthest ~30%

    is_short = L < L_thresh
    is_front = d > d_front
    is_back  = d < d_back

    # drop short arrows both in front & back
    keep = ~(is_short & (is_front | is_back))

    x0 = x0[keep]
    y0 = y0[keep]
    z0 = z0[keep]
    dx = dx[keep]
    dy = dy[keep]
    dz = dz[keep]

    # ------------------------------------------------------------
    # plot arrows
    # ------------------------------------------------------------
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")

    for (xs, ys, zs, ddx, ddy, ddz) in zip(x0, y0, z0, dx, dy, dz):
        arrow = Arrow3D(
            [xs, xs + ddx], [ys, ys + ddy], [zs, zs + ddz],
            mutation_scale=12,
            lw=2.2,
            arrowstyle="-|>",
            color="C0",
        )
        ax.add_artist(arrow)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.set_xlim(-grid_lim, grid_lim)
    ax.set_ylim(-grid_lim, grid_lim)
    ax.set_zlim(-grid_lim, grid_lim)
    ax.view_init(elev=elev, azim=azim)

    plt.tight_layout()

    return ax

