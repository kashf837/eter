#!/usr/bin/env python3
"""Convert STEP B_SPLINE_SURFACE to .surf (supports inline and entity formats)."""
import sys, os, re

def parse_cp_grid(grid_text):
    """Parse nested CP grid like ((#1,#2),(#3,#4)) into list of row strings."""
    inner = grid_text.strip()
    if inner.startswith('('): inner = inner[1:]
    if inner.endswith(')'): inner = inner[:-1]
    rows, depth, cur = [], 0, ''
    for ch in inner:
        if ch == '(': depth += 1
        elif ch == ')':
            depth -= 1
            if depth == 0: rows.append(cur.strip()); cur = ''; continue
        if depth == 1: cur += ch
    return rows


def parse_step_file(path):
    """Read STEP file, return (entities dict, list of surface data dicts)."""
    with open(path) as f:
        text = f.read()

    # Collect all CARTESIAN_POINT entities
    cart_points = {}
    for m in re.finditer(r'#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*\'[^\']*\'\s*,\s*\(\s*([^)]+)\s*\)\s*\)\s*;', text):
        cid = int(m.group(1))
        xyz = [float(v.strip()) for v in m.group(2).split(',')]
        while len(xyz) < 3: xyz.append(0.0)  # pad 1D/2D to 3D
        cart_points[cid] = xyz

    # Parse DIRECTION entities (needed for AXIS2_PLACEMENT_3D → PLANE)
    directions = {}
    for m in re.finditer(r'#(\d+)\s*=\s*DIRECTION\s*\(\s*\'[^\']*\'\s*,\s*\(\s*([^)]+)\s*\)\s*\)\s*;', text):
        did = int(m.group(1))
        xyz = [float(v.strip()) for v in m.group(2).split(',')]
        while len(xyz) < 3: xyz.append(0.0)
        # Normalize (STEP directions may or may not be unit-length)
        mag = (xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]) ** 0.5
        if mag > 0: xyz = [v/mag for v in xyz]
        directions[did] = xyz

    # Parse AXIS2_PLACEMENT_3D entities (origin, Z-axis/normal, X-axis/ref_dir)
    axises = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*AXIS2_PLACEMENT_3D\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*\)\s*;', text):
        aid = int(m.group(1))
        oid, zid, xid = int(m.group(2)), int(m.group(3)), int(m.group(4))
        if oid in cart_points and zid in directions and xid in directions:
            axises[aid] = (cart_points[oid], directions[zid], directions[xid])

    surfaces = []
    plane_surfaces = []  # collected separately to not interfere with Format B fallback
    skipped_fmts = {}  # track skipped surface types for diagnostics

    # Compute model bounding box from CARTESIAN_POINTs for default plane sizing
    bbox_min = [float('inf')] * 3
    bbox_max = [float('-inf')] * 3
    for cid, xyz in cart_points.items():
        for i in range(3):
            if xyz[i] < bbox_min[i]: bbox_min[i] = xyz[i]
            if xyz[i] > bbox_max[i]: bbox_max[i] = xyz[i]
    from math import sqrt
    bbox_diag = sqrt(sum((bbox_max[i] - bbox_min[i])**2 for i in range(3)))
    plane_half = bbox_diag * 0.002  # 0.2% of model diagonal per half-extent

    # --- PLANE surfaces → degree-1×1 B-spline patches ---
    for m in re.finditer(r'#(\d+)\s*=\s*PLANE\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*\)\s*;', text):
        aid = int(m.group(2))
        if aid not in axises: continue
        origin, normal, x_dir = axises[aid]
        # Compute orthonormal basis on the plane
        ux, uy, uz = x_dir  # U axis = X direction (already normalized)
        # V axis = cross(normal, x_dir)
        n = normal
        vx = n[1]*uz - n[2]*uy
        vy = n[2]*ux - n[0]*uz
        vz = n[0]*uy - n[1]*ux
        # 4 CPs for a rectangular plane patch centered at origin
        def corner(su, sv):
            return [origin[0] + su*plane_half*ux + sv*plane_half*vx,
                    origin[1] + su*plane_half*uy + sv*plane_half*vy,
                    origin[2] + su*plane_half*uz + sv*plane_half*vz]
        surf = {
            'p_u': 1, 'p_v': 1,
            'n_u': 1, 'n_v': 1,
            'U': [0.0, 0.0, 1.0, 1.0],
            'V': [0.0, 0.0, 1.0, 1.0],
            'cp_grid': [
                [corner(-1, -1), corner(-1, 1)],
                [corner( 1, -1), corner( 1, 1)],
            ]
        }
        plane_surfaces.append(surf)

    # --- Format A: standalone B_SPLINE_SURFACE_WITH_KNOTS entities ---
    for m in re.finditer(r'#(\d+)\s*=\s*B_SPLINE_SURFACE_WITH_KNOTS\s*\((.*?)\)\s*;', text, re.DOTALL):
        try:
            surf = parse_standalone_surface(m.group(2), cart_points)
            if surf:
                surfaces.append(surf)
        except:
            pass

    # --- Format B: inline (BOUNDED_SURFACE wrapping B_SPLINE_SURFACE) ---
    if not surfaces:
        # Find all inline surfaces:  B_SPLINE_SURFACE ( pu, pv, ( CP grid ) ...
        # followed by B_SPLINE_SURFACE_WITH_KNOTS
        for m in re.finditer(
            r'B_SPLINE_SURFACE\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*\(((?:[^()]*\([^()]*\)[^()]*)+)\)\s*,'  # degrees + CP grid
            r'(.*?)'  # middle section (.UNSPECIFIED., .F., etc.)
            r'B_SPLINE_SURFACE_WITH_KNOTS\s*\(((?:[^()]*\([^()]*\)[^()]*)+)\)',  # knot data
            text, re.DOTALL
        ):
            try:
                p_u, p_v = int(m.group(1)), int(m.group(2))
                cp_grid_raw = m.group(3)
                knot_raw = m.group(5)

                # Parse CP grid rows with depth-based parser
                cp_grid = []
                for row_text in parse_cp_grid(cp_grid_raw):
                    clean = row_text.strip().lstrip('(')
                    ids = [int(s.strip().lstrip('#')) for s in clean.split(',') if s.strip()]
                    row_pts = [cart_points[cid] for cid in ids if cid in cart_points]
                    if row_pts: cp_grid.append(row_pts)

                if len(cp_grid) < 2:
                    continue

                # Parse knot data: (u_mults), (v_mults), (u_knots), (v_knots)
                lists = re.findall(r'\(([0-9eE.\-+,\s]+)\)', knot_raw)
                if len(lists) >= 4:
                    u_mult  = [int(float(v.strip())) for v in lists[0].split(',') if v.strip()]
                    v_mult  = [int(float(v.strip())) for v in lists[1].split(',') if v.strip()]
                    u_knots = [float(v.strip()) for v in lists[2].split(',') if v.strip()]
                    v_knots = [float(v.strip()) for v in lists[3].split(',') if v.strip()]

                    U = []; V = []
                    for k, m in zip(u_knots, u_mult): U.extend([k]*m)
                    for k, m in zip(v_knots, v_mult): V.extend([k]*m)

                    n_v = max(len(r) for r in cp_grid) - 1 if cp_grid else 0
                    surf = {
                        'p_u': p_u, 'p_v': p_v,
                        'n_u': len(cp_grid) - 1,
                        'n_v': n_v,
                        'U': U, 'V': V,
                        'cp_grid': cp_grid
                    }
                    surfaces.append(surf)
            except Exception as e:
                pass

    # Prepend plane surfaces (after Format A/B detection to preserve B-spline fallback)
    surfaces = plane_surfaces + surfaces

    # --- Diagnostic: count all surface types in the STEP file ---
    all_surf_types = {}
    for m in re.finditer(r'#\d+\s*=\s*(\w+)\s*\(', text):
        t = m.group(1)
        if 'SURFACE' in t or 'SURF' in t or t in ('PLANE','CYLINDRICAL','SPHERICAL','CONICAL','TOROIDAL',
            'EXTRUSION','REVOLUTION','SWEPT','BLEND','OFFSET','TRIMMED'):
            all_surf_types[t] = all_surf_types.get(t, 0) + 1
    if all_surf_types:
        print(f"STEP surface types found: {dict(sorted(all_surf_types.items()))}")
        bsp_total = all_surf_types.get('B_SPLINE_SURFACE', 0) + all_surf_types.get('B_SPLINE_SURFACE_WITH_KNOTS', 0)
        plane_total = all_surf_types.get('PLANE', 0)
        non_bsp = sum(v for k,v in all_surf_types.items()
                      if k not in ('B_SPLINE_SURFACE','B_SPLINE_SURFACE_WITH_KNOTS','PLANE'))
        if non_bsp > 0:
            print(f"WARNING: {non_bsp} non-B-spline/non-PLANE surfaces SKIPPED ({len(surfaces)} surfaces loaded)")
        else:
            print(f"Loaded {len(surfaces)} surfaces ({bsp_total} B-splines + {plane_total} planes)")

    return surfaces


def parse_standalone_surface(body, cart_points):
    """Parse Format A body."""
    parts = split_body(body)
    p_u = int(parts[1].strip())
    p_v = int(parts[2].strip())

    # Parse CP grid with depth-based parser
    cp_grid = []
    for row_text in parse_cp_grid(parts[3]):
        # Strip leading '(' that may prefix the first ref
        clean = row_text.strip().lstrip('(')
        ids = [int(s.strip().lstrip('#')) for s in clean.split(',') if s.strip().startswith('#')]
        row_pts = [cart_points[cid] for cid in ids if cid in cart_points]
        if row_pts: cp_grid.append(row_pts)

    knot_section = ','.join(parts[4:])
    lists = re.findall(r'\(([0-9eE.\-+,\s]+)\)', knot_section)
    if len(lists) < 4:
        return None
    u_mult  = [int(float(v.strip())) for v in lists[0].split(',') if v.strip()]
    v_mult  = [int(float(v.strip())) for v in lists[1].split(',') if v.strip()]
    u_knots = [float(v.strip()) for v in lists[2].split(',') if v.strip()]
    v_knots = [float(v.strip()) for v in lists[3].split(',') if v.strip()]

    U, V = [], []
    for k, m in zip(u_knots, u_mult): U.extend([k] * m)
    for k, m in zip(v_knots, v_mult): V.extend([k] * m)

    n_v = max(len(row) for row in cp_grid) - 1 if cp_grid else 0
    return {'p_u': p_u, 'p_v': p_v, 'n_u': len(cp_grid)-1, 'n_v': n_v,
            'U': U, 'V': V, 'cp_grid': cp_grid}


def split_body(body):
    parts, depth, cur = [], 0, ''
    for ch in body:
        if ch == '(': depth += 1
        elif ch == ')': depth -= 1
        if ch == ',' and depth == 0:
            parts.append(cur); cur = ''
        else: cur += ch
    if cur.strip(): parts.append(cur)
    return parts


def write_surf(path, s):
    with open(path, 'w') as f:
        f.write(f"# STEP surface\n{s['p_u']} {s['p_v']}  {s['n_u']} {s['n_v']}\n")
        f.write(' '.join(f"{v:.10f}" for v in s['U']) + '\n')
        f.write(' '.join(f"{v:.10f}" for v in s['V']) + '\n')
        for i in range(s['n_u'] + 1):
            for j in range(s['n_v'] + 1):
                xyz = s['cp_grid'][i][j]
                f.write(f"{xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f} 1.0\n")
    # print(f"Wrote {path}: {s['n_u']+1}x{s['n_v']+1} CPs, order {s['p_u']}x{s['p_v']}")


def main():
    all_flag = '--all' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--all']
    if len(args) < 1:
        print("Usage: python step2surf.py [--all] <input.STEP> [output]")
        return

    infile = args[0]

    # Safety: refuse to overwrite the input file
    if not all_flag and len(args) < 2:
        print("ERROR: single-file mode requires an output path (to avoid overwriting input)")
        print("Usage: python step2surf.py [--all] <input.STEP> <output.surf>")
        return

    surfaces = parse_step_file(infile)

    if not surfaces:
        print("No surfaces found")
        return

    print(f"Found {len(surfaces)} surfaces")

    if all_flag:
        base = args[1] if len(args) > 1 else re.sub(r'(?i)\.(STEP|STP)$', '', infile)
        os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
        for idx, s in enumerate(surfaces):
            write_surf(f"{base}_{idx:04d}.surf", s)
    else:
        out = args[1]
        write_surf(out, surfaces[0])


if __name__ == '__main__':
    main()
