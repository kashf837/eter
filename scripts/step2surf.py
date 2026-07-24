#!/usr/bin/env python3
"""Convert STEP AP214 B_SPLINE_SURFACE_WITH_KNOTS to .surf format."""
import sys, re

def parse_step(path):
    """Read STEP file, return dict of entities and the first surface entity ID."""
    with open(path) as f:
        text = f.read()

    entities = {}
    surface_id = None

    # Find all entities:  #ID = TYPE ( ... ) ;
    for m in re.finditer(r'#(\d+)\s*=\s*(\w+)\s*\((.*?)\)\s*;', text, re.DOTALL):
        eid = int(m.group(1))
        etype = m.group(2)
        body = m.group(3)
        entities[eid] = (etype, body)
        if etype == 'B_SPLINE_SURFACE_WITH_KNOTS' and surface_id is None:
            surface_id = eid  # take the first one

    return entities, surface_id


def parse_surface(entities, sid):
    """Parse B_SPLINE_SURFACE_WITH_KNOTS and return .surf-compatible data."""
    _, body = entities[sid]

    # Split body at commas, but respect parentheses nesting
    parts = split_body(body)

    # parts[0]: name ('NONE')
    # parts[1]: u_degree
    p_u = int(parts[1].strip())
    # parts[2]: v_degree
    p_v = int(parts[2].strip())

    # parts[3]: control point grid
    cp_grid_raw = parts[3]

    # Parse the CP grid:  ( ( #id, #id, ... ), ( ... ), ... )
    # Each row is ( #id #id ... )
    rows = re.findall(r'\(([^)]+)\)', cp_grid_raw)
    cp_ids = []
    for row in rows:
        ids = [int(s.strip().lstrip('#')) for s in row.split(',') if s.strip().startswith('#')]
        cp_ids.append(ids)

    n_u = len(cp_ids) - 1
    n_v = len(cp_ids[0]) - 1

    # Resolve control points
    cp_grid = []
    for row in cp_ids:
        cp_row = []
        for cid in row:
            _, cp_body = entities[cid]
            # CARTESIAN_POINT ( 'NONE', ( x, y, z ) )
            m = re.search(r'\(([^)]+)\)', cp_body)
            if m:
                xyz = [float(v.strip()) for v in m.group(1).split(',')]
                cp_row.append(xyz)
        cp_grid.append(cp_row)

    # Skip ahead to knot data
    # After CP grid: .UNSPECIFIED., .F., .F., .F.,
    # Then: ( u_mult ), ( v_mult ), ( u_knots ), ( v_knots )
    idx = 4  # skip name, pu, pv, cp_grid

    # Find knot multiplicity arrays
    u_mult, v_mult, u_knots, v_knots = None, None, None, None
    knot_section = ','.join(parts[idx:])

    # Find parenthesized lists
    lists = re.findall(r'\(([0-9.,\s]+)\)', knot_section)
    if len(lists) >= 4:
        u_mult  = [int(v.strip()) for v in lists[0].split(',') if v.strip()]
        v_mult  = [int(v.strip()) for v in lists[1].split(',') if v.strip()]
        u_knots = [float(v.strip()) for v in lists[2].split(',') if v.strip()]
        v_knots = [float(v.strip()) for v in lists[3].split(',') if v.strip()]

    # Expand knot vectors
    U = expand_knots(u_knots, u_mult)
    V = expand_knots(v_knots, v_mult)

    return p_u, p_v, n_u, n_v, U, V, cp_grid


def split_body(body):
    """Split body at top-level commas (not inside parentheses)."""
    parts = []
    depth = 0
    current = ''
    for ch in body:
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        if ch == ',' and depth == 0:
            parts.append(current)
            current = ''
        else:
            current += ch
    if current.strip():
        parts.append(current)
    return parts


def expand_knots(knots, mults):
    """Expand (knot_value, multiplicity) pairs into full knot vector."""
    result = []
    for k, m in zip(knots, mults):
        result.extend([k] * m)
    return result


def write_surf(path, p_u, p_v, n_u, n_v, U, V, cp_grid):
    """Write .surf file."""
    with open(path, 'w') as f:
        f.write(f"# Converted from STEP\n")
        f.write(f"{p_u} {p_v}  {n_u} {n_v}\n")
        f.write(' '.join(f"{v:.10f}" for v in U) + '\n')
        f.write(' '.join(f"{v:.10f}" for v in V) + '\n')
        for i in range(n_u + 1):
            for j in range(n_v + 1):
                xyz = cp_grid[i][j]
                f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} 1.0\n")
    print(f"Wrote {path}: {n_u+1}x{n_v+1} CPs, order {p_u}x{p_v}")


def main():
    if len(sys.argv) < 2:
        print("Usage: python step2surf.py <input.STEP> [output.surf]")
        return

    infile = sys.argv[1]
    outfile = sys.argv[2] if len(sys.argv) > 2 else infile.replace('.STEP', '.surf').replace('.step', '.surf')

    entities, sid = parse_step(infile)
    if sid is None:
        print("No B_SPLINE_SURFACE_WITH_KNOTS found")
        return

    try:
        p_u, p_v, n_u, n_v, U, V, cp_grid = parse_surface(entities, sid)
        write_surf(outfile, p_u, p_v, n_u, n_v, U, V, cp_grid)
    except Exception as e:
        print(f"Parse failed: {e}")


if __name__ == '__main__':
    main()
