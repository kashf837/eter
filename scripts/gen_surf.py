#!/usr/bin/env python3
"""Generate NURBS/B-spline surface control point files (.surf format)."""

import sys, math, argparse

def make_knot_vector(n_cp, p):
    """Clamped knot vector for n_cp CPs (index, so n_cp+1 CPs) and degree p."""
    internal = n_cp - p
    knots = [0.0] * (p + 1)
    for i in range(1, internal + 1):
        knots.append(float(i) / (internal + 1))
    knots.extend([1.0] * (p + 1))
    return knots

def write_surf(path, p_u, p_v, cp_grid, weights=None):
    """Write .surf file.
    cp_grid: 2D list of (x,y,z) or (x,y,z,w) tuples, shape (n_u+1) x (n_v+1)
    weights: optional 2D list of weights; if None, all w=1
    """
    n_u = len(cp_grid) - 1
    n_v = len(cp_grid[0]) - 1

    U = make_knot_vector(n_u, p_u)
    V = make_knot_vector(n_v, p_v)

    with open(path, 'w') as f:
        f.write(f"# Generated surface\n")
        f.write(f"{p_u} {p_v}  {n_u} {n_v}\n")
        f.write(' '.join(f"{v:.1f}" for v in U) + '\n')
        f.write(' '.join(f"{v:.1f}" for v in V) + '\n')
        for i in range(n_u + 1):
            for j in range(n_v + 1):
                x, y, z = cp_grid[i][j][0], cp_grid[i][j][1], cp_grid[i][j][2]
                w = weights[i][j] if weights else 1.0
                f.write(f"{x:.6f} {y:.6f} {z:.6f} {w:.6f}\n")
    print(f"Wrote {path}: {n_u+1}x{n_v+1} CPs, order {p_u}x{p_v}")


# ========== Shape generators ==========

def gen_flat(p_u=2, p_v=2, n_u=3, n_v=3):
    """A flat square [0,4]x[0,4] at z=0."""
    grid = [[(4.0 * i / n_u, 4.0 * j / n_v, 0.0) for j in range(n_v + 1)] for i in range(n_u + 1)]
    return grid, None

def gen_bulge(p_u=2, p_v=2, n_u=3, n_v=3):
    """A bulging pillow surface (diamond contour)."""
    grid = []
    for i in range(n_u + 1):
        row = []
        for j in range(n_v + 1):
            x = float(i) / n_u
            y = float(j) / n_v
            z = 1.0 - 2.0 * abs(x - 0.5) - 2.0 * abs(y - 0.5)
            z = max(0.0, z)
            row.append((x, y, z))
        grid.append(row)
    return grid, None

def gen_dome(p_u=2, p_v=2, n_u=3, n_v=3):
    """A circular dome: z = 1 - 4*r^2, r = distance from center."""
    grid = []
    for i in range(n_u + 1):
        row = []
        for j in range(n_v + 1):
            x = float(i) / n_u
            y = float(j) / n_v
            r2 = (x - 0.5)**2 + (y - 0.5)**2
            z = 1.0 - 4.0 * r2
            z = max(0.0, z)
            row.append((x, y, z))
        grid.append(row)
    return grid, None

def gen_ripple(p_u=2, p_v=2, n_u=5, n_v=5):
    """Concentric water ripples, decaying with distance from center."""
    import math
    grid = []
    for i in range(n_u + 1):
        row = []
        for j in range(n_v + 1):
            x = float(i) / n_u
            y = float(j) / n_v
            r = math.sqrt((x - 0.5)**2 + (y - 0.5)**2)
            z = 0.25 * math.sin(r * 15.0) * math.exp(-r * 2.5)
            row.append((x, y, z))
        grid.append(row)
    return grid, None

def gen_bumps(p_u=2, p_v=2, n_u=5, n_v=5):
    """Three overlapping Gaussian bumps at random-ish positions."""
    import math
    bumps = [(0.3, 0.3, 0.08), (0.7, 0.6, 0.06), (0.5, 0.2, 0.07)]
    grid = []
    for i in range(n_u + 1):
        row = []
        for j in range(n_v + 1):
            x = float(i) / n_u
            y = float(j) / n_v
            z = 0.0
            for bx, by, bw in bumps:
                r2 = ((x - bx)**2 + (y - by)**2) / (bw * bw)
                z += math.exp(-r2)
            z = (z - 0.3) * 1.2  # normalize
            row.append((x, y, z))
        grid.append(row)
    return grid, None

def gen_saddle(p_u=2, p_v=2, n_u=3, n_v=3):
    """A saddle surface z = (x-0.5)*(y-0.5)."""
    grid = []
    for i in range(n_u + 1):
        row = []
        for j in range(n_v + 1):
            x = float(i) / n_u
            y = float(j) / n_v
            z = 4.0 * (x - 0.5) * (y - 0.5)
            row.append((x, y, z))
        grid.append(row)
    return grid, None

def gen_wave(p_u=2, p_v=2, n_u=5, n_v=5):
    """A wavy surface z = sin(2*pi*x) * cos(2*pi*y)."""
    grid = []
    for i in range(n_u + 1):
        row = []
        for j in range(n_v + 1):
            x = float(i) / n_u
            y = float(j) / n_v
            z = 0.3 * math.sin(2 * math.pi * x) * math.cos(2 * math.pi * y)
            row.append((x, y, z))
        grid.append(row)
    return grid, None

def gen_sphere(p_u=2, p_v=2, n_u=4, n_v=4):
    """Approximate octant of a unit sphere (x>=0, y>=0, z>=0)."""
    # Sample sphere: cp_grid[i][j] = point on sphere at that parameter location
    # This is a sampled-data CP set — NOT an exact sphere, but looks spherical
    grid = []
    for i in range(n_u + 1):
        row = []
        u = i * math.pi / (2 * n_u)
        for j in range(n_v + 1):
            v = j * math.pi / (2 * n_v)
            x = math.sin(v) * math.cos(u)
            y = math.sin(v) * math.sin(u)
            z = math.cos(v)
            row.append((x, y, z))
        grid.append(row)
    return grid, None

def gen_cylinder(p_u=2, p_v=2, n_u=3, n_v=3):
    """Quarter-cylinder approximation along z-axis."""
    grid = []
    for i in range(n_u + 1):
        row = []
        z = float(i) / n_u
        for j in range(n_v + 1):
            theta = j * math.pi / (2 * n_v)
            x = math.cos(theta)
            y = math.sin(theta)
            row.append((x, y, z))
        grid.append(row)
    return grid, None


# ========== CLI ==========

SHAPES = {
    'flat':     gen_flat,
    'bulge':    gen_bulge,
    'dome':     gen_dome,
    'ripple':   gen_ripple,
    'bumps':    gen_bumps,
    'saddle':   gen_saddle,
    'wave':     gen_wave,
    'sphere':   gen_sphere,
    'cylinder': gen_cylinder,
}

def main():
    parser = argparse.ArgumentParser(description='Generate NURBS .surf files')
    parser.add_argument('--shape', choices=list(SHAPES.keys()), default='bulge',
                        help='Shape type')
    parser.add_argument('--out', default='data/gen.surf',
                        help='Output file path')
    parser.add_argument('--order', type=int, nargs=2, default=[2, 2],
                        metavar=('PU', 'PV'), help='Degree in u and v (default: 2 2)')
    parser.add_argument('--grid', type=int, nargs=2, default=[3, 3],
                        metavar=('NU', 'NV'), help='CP grid size minus 1 (default: 3 3 = 4x4 CPs)')
    args = parser.parse_args()

    p_u, p_v = args.order
    n_u, n_v = args.grid

    grid, weights = SHAPES[args.shape](p_u, p_v, n_u, n_v)
    write_surf(args.out, p_u, p_v, grid, weights)


if __name__ == '__main__':
    main()
