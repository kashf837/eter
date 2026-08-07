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

    # Compute model bounding box from CARTESIAN_POINTs for fallback plane sizing
    bbox_min = [float('inf')] * 3
    bbox_max = [float('-inf')] * 3
    for cid, xyz in cart_points.items():
        for i in range(3):
            if xyz[i] < bbox_min[i]: bbox_min[i] = xyz[i]
            if xyz[i] > bbox_max[i]: bbox_max[i] = xyz[i]
    from math import sqrt
    bbox_diag = sqrt(sum((bbox_max[i] - bbox_min[i])**2 for i in range(3)))
    fallback_half = bbox_diag * 0.0005  # tiny fallback if trimming chain fails

    # --- Parse trimming boundary chain for PLANE surfaces ---
    # VERTEX_POINT: id → cartesian_point_id
    vertex_points = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*VERTEX_POINT\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*\)\s*;', text):
        vertex_points[int(m.group(1))] = int(m.group(2))

    # EDGE_CURVE: id → (vertex_point_id_1, vertex_point_id_2)
    edge_curves = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*EDGE_CURVE\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,', text):
        edge_curves[int(m.group(1))] = (int(m.group(2)), int(m.group(3)))

    # ORIENTED_EDGE: id → edge_curve_id
    oriented_edges = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*ORIENTED_EDGE\s*\(\s*\'[^\']*\'\s*,\s*\*\s*,\s*\*\s*,\s*#(\d+)\s*,', text):
        oriented_edges[int(m.group(1))] = int(m.group(2))

    # EDGE_LOOP: id → [oriented_edge_ids]
    edge_loops = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*EDGE_LOOP\s*\(\s*\'[^\']*\'\s*,\s*\(([^)]+)\)', text):
        el_id = int(m.group(1))
        oe_list = [int(s.strip().lstrip('#')) for s in m.group(2).split(',') if s.strip()]
        edge_loops[el_id] = oe_list

    # FACE_OUTER_BOUND: id → edge_loop_id
    face_outer_bounds = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*FACE_OUTER_BOUND\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,', text):
        face_outer_bounds[int(m.group(1))] = int(m.group(2))

    # ADVANCED_FACE: surface_id → [face_outer_bound_ids]
    surface_to_bounds = {}
    for m in re.finditer(
        r'#\d+\s*=\s*ADVANCED_FACE\s*\(\s*\'[^\']*\'\s*,\s*\(([^)]+)\)\s*,\s*#(\d+)\s*,', text):
        surface_id = int(m.group(2))
        bnds = [int(s.strip().lstrip('#')) for s in m.group(1).split(',') if s.strip()]
        surface_to_bounds[surface_id] = bnds

    # --- PLANE surfaces with trimming boundary → degree-1×1 B-spline patches ---
    from math import sqrt as msqrt
    for m in re.finditer(r'#(\d+)\s*=\s*PLANE\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*\)\s*;', text):
        plane_id = int(m.group(1))
        aid = int(m.group(2))
        if aid not in axises: continue

        origin, normal, x_dir = axises[aid]
        ux, uy, uz = x_dir
        n = normal
        vx = n[1]*uz - n[2]*uy
        vy = n[2]*ux - n[0]*uz
        vz = n[0]*uy - n[1]*ux

        half_u, half_v = fallback_half, fallback_half
        center = origin[:]  # may be shifted to vertex centroid

        # Trace trimming boundary: PLANE → ADVANCED_FACE → FACE_OUTER_BOUND → EDGE_LOOP → VERTEX
        if plane_id in surface_to_bounds:
            vertices = []
            for bound_id in surface_to_bounds[plane_id]:
                if bound_id not in face_outer_bounds: continue
                el_id = face_outer_bounds[bound_id]
                if el_id not in edge_loops: continue
                for oe_id in edge_loops[el_id]:
                    if oe_id not in oriented_edges: continue
                    ec_id = oriented_edges[oe_id]
                    if ec_id not in edge_curves: continue
                    for vp_id in edge_curves[ec_id]:
                        cp_id = vertex_points.get(vp_id)
                        if cp_id and cp_id in cart_points:
                            vertices.append(cart_points[cp_id])

            # Deduplicate (multiple edges share vertices)
            seen = set()
            unique_vtx = []
            for vtx in vertices:
                key = (round(vtx[0], 6), round(vtx[1], 6), round(vtx[2], 6))
                if key not in seen:
                    seen.add(key)
                    unique_vtx.append(vtx)

            if len(unique_vtx) >= 3:
                # Project vertices to plane local 2D
                u_vals, v_vals = [], []
                for vtx in unique_vtx:
                    dx, dy, dz = vtx[0] - origin[0], vtx[1] - origin[1], vtx[2] - origin[2]
                    u_vals.append(dx*ux + dy*uy + dz*uz)
                    v_vals.append(dx*vx + dy*vy + dz*vz)
                u_min, u_max = min(u_vals), max(u_vals)
                v_min, v_max = min(v_vals), max(v_vals)

                # Center the patch at the trimmed region's centroid
                cu, cv = (u_min + u_max) * 0.5, (v_min + v_max) * 0.5
                center = [origin[0] + cu*ux + cv*vx,
                          origin[1] + cu*uy + cv*vy,
                          origin[2] + cu*uz + cv*vz]

                # Slight margin to ensure full coverage
                margin = max(u_max - u_min, v_max - v_min, 1.0) * 0.02 + 0.1
                half_u = (u_max - u_min) * 0.5 + margin
                half_v = (v_max - v_min) * 0.5 + margin

        def corner(su, sv):
            return [center[0] + su*half_u*ux + sv*half_v*vx,
                    center[1] + su*half_u*uy + sv*half_v*vy,
                    center[2] + su*half_u*uz + sv*half_v*vz]

        plane_surfaces.append({
            'p_u': 1, 'p_v': 1,
            'n_u': 1, 'n_v': 1,
            'U': [0.0, 0.0, 1.0, 1.0],
            'V': [0.0, 0.0, 1.0, 1.0],
            'cp_grid': [
                [corner(-1, -1), corner(-1, 1)],
                [corner( 1, -1), corner( 1, 1)],
            ]
        })

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
