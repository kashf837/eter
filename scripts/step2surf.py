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

    surfaces = []
    skipped_fmts = {}  # track skipped surface types for diagnostics

    # ================================================================
    # PLANE → trimmed triangle patches (boundary chain + triangulation)
    # ================================================================

    # Parse DIRECTION entities (for AXIS2_PLACEMENT_3D)
    directions = {}
    for m in re.finditer(r'#(\d+)\s*=\s*DIRECTION\s*\(\s*\'[^\']*\'\s*,\s*\(\s*([^)]+)\s*\)\s*\)\s*;', text):
        xyz = [float(v.strip()) for v in m.group(2).split(',')]
        while len(xyz) < 3: xyz.append(0.0)
        mag = (xyz[0]*xyz[0] + xyz[1]*xyz[1] + xyz[2]*xyz[2]) ** 0.5
        if mag > 0: xyz = [v/mag for v in xyz]
        directions[int(m.group(1))] = xyz

    # Parse AXIS2_PLACEMENT_3D entities
    axises = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*AXIS2_PLACEMENT_3D\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*\)\s*;', text):
        aid, oid, zid, xid = int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4))
        if oid in cart_points and zid in directions and xid in directions:
            axises[aid] = (cart_points[oid], directions[zid], directions[xid])

    # Boundary chain: VERTEX_POINT → EDGE_CURVE → ORIENTED_EDGE → EDGE_LOOP → FACE_OUTER_BOUND
    vertex_points = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*VERTEX_POINT\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*\)\s*;', text):
        vertex_points[int(m.group(1))] = int(m.group(2))

    # EDGE_CURVE: id → (vp1, vp2, curve_geom_id)
    edge_curves = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*EDGE_CURVE\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,\s*#(\d+)\s*,', text):
        edge_curves[int(m.group(1))] = (int(m.group(2)), int(m.group(3)), int(m.group(4)))

    # --- Parse B-spline curve geometries for edge sampling ---
    # B_SPLINE_CURVE_WITH_KNOTS: id → (degree, cps, knots, domain_min, domain_max)
    bsp_curves = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*B_SPLINE_CURVE_WITH_KNOTS\s*\((.*?)\)\s*;', text, re.DOTALL):
        cid = int(m.group(1))
        body = m.group(2)
        # Split body at top-level commas
        parts, depth, cur = [], 0, ''
        for ch in body:
            if ch == '(': depth += 1
            elif ch == ')': depth -= 1
            if ch == ',' and depth == 0:
                parts.append(cur.strip()); cur = ''
            else: cur += ch
        if cur.strip(): parts.append(cur.strip())
        try:
            # parts[0] = name, parts[1] = degree
            deg = int(parts[1])
            # parts[2] = CP list: ((#cp1,#cp2,...),(#cp3,...),...)
            cp_text = parts[2]
            cp_ids = [int(s.strip().lstrip('#')) for s in cp_text.split(',') if s.strip().lstrip('#').isdigit()]
            cps = [cart_points[cid_] for cid_ in cp_ids if cid_ in cart_points]
            if len(cps) < 2: continue
            # Find knot data: look for the multiplicity and knot lists
            # After CPs: .UNSPECIFIED., .F., .F., (mults), (knots)
            knot_mults = []
            knot_vals = []
            for p in parts[3:]:
                p = p.strip()
                if p.startswith('(') and not knot_mults:
                    knot_mults = [int(float(x.strip())) for x in p.strip('()').split(',') if x.strip()]
                elif p.startswith('(') and knot_mults and not knot_vals:
                    knot_vals = [float(x.strip()) for x in p.strip('()').split(',') if x.strip()]
            if knot_mults and knot_vals:
                # Build full knot vector
                knots = []
                for kval, mult in zip(knot_vals, knot_mults):
                    knots.extend([kval] * mult)
                domain_min = knots[deg]
                domain_max = knots[len(cps)]
                bsp_curves[cid] = (deg, cps, knots, domain_min, domain_max)
        except:
            pass

    # B-spline curve evaluation (Cox-de Boor)
    def eval_curve(curve, t):
        deg, cps, knots, _, _ = curve
        n = len(cps) - 1
        # Clamp t to domain
        t = max(knots[deg], min(t, knots[n + 1]))
        # Find knot span
        span = deg
        for i in range(deg, n + 1):
            if knots[i] <= t < knots[i + 1]:
                span = i
                break
        if t >= knots[n + 1]:
            span = n
        # Basis functions
        N = [0.0] * (deg + 1)
        left = [0.0] * (deg + 1)
        right = [0.0] * (deg + 1)
        N[0] = 1.0
        for j in range(1, deg + 1):
            left[j] = t - knots[span + 1 - j]
            right[j] = knots[span + j] - t
            saved = 0.0
            for r in range(j):
                temp = N[r] / (right[r + 1] + left[j - r]) if (right[r + 1] + left[j - r]) > 1e-15 else 0.0
                N[r] = saved + right[r + 1] * temp
                saved = left[j - r] * temp
            N[j] = saved
        # Evaluate
        pt = [0.0, 0.0, 0.0]
        for i in range(deg + 1):
            cp = cps[span - deg + i]
            w = N[i]
            for d in range(3):
                pt[d] += w * cp[d]
        return pt

    # ORIENTED_EDGE: id → (edge_curve_id, orientation)  orientation=True means forward
    oriented_edges = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*ORIENTED_EDGE\s*\(\s*\'[^\']*\'\s*,\s*\*\s*,\s*\*\s*,\s*#(\d+)\s*,\s*(\.\w\.)\s*\)', text):
        oriented_edges[int(m.group(1))] = (int(m.group(2)), m.group(3) == '.T.')

    # EDGE_LOOP: id → [oriented_edge_ids]
    edge_loops = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*EDGE_LOOP\s*\(\s*\'[^\']*\'\s*,\s*\(([^)]+)\)', text):
        el_id = int(m.group(1))
        edge_loops[el_id] = [int(s.strip().lstrip('#')) for s in m.group(2).split(',') if s.strip()]

    # FACE_OUTER_BOUND: id → edge_loop_id
    face_outer_bounds = {}
    for m in re.finditer(
        r'#(\d+)\s*=\s*FACE_OUTER_BOUND\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*,', text):
        face_outer_bounds[int(m.group(1))] = int(m.group(2))

    # ADVANCED_FACE: surface_id → [bound_ids]
    surface_to_bounds = {}
    for m in re.finditer(
        r'#\d+\s*=\s*ADVANCED_FACE\s*\(\s*\'[^\']*\'\s*,\s*\(([^)]+)\)\s*,\s*#(\d+)\s*,', text):
        surface_id = int(m.group(2))
        surface_to_bounds[surface_id] = [int(s.strip().lstrip('#'))
            for s in m.group(1).split(',') if s.strip()]

    # For each PLANE: trace boundary → ordered vertices → triangulate → degree-elevate
    # Degree elevation coefficients: degree 1 → 3 (for Bézier basis)
    elev = [
        [1.0, 0.0],
        [2.0/3.0, 1.0/3.0],
        [1.0/3.0, 2.0/3.0],
        [0.0, 1.0],
    ]

    def elev_cp(P, k, l):
        """Elevate bilinear CP grid P[2][2] to bicubic at (k,l), k,l=0..3"""
        r = [0.0, 0.0, 0.0, 1.0]
        for i in range(2):
            for j in range(2):
                w = elev[k][i] * elev[l][j]
                for d in range(3):
                    r[d] += w * P[i][j][d]
        return r

    plane_tri_count = 0
    for m in re.finditer(r'#(\d+)\s*=\s*PLANE\s*\(\s*\'[^\']*\'\s*,\s*#(\d+)\s*\)\s*;', text):
        plane_id = int(m.group(1))
        aid = int(m.group(2))
        if aid not in axises: continue
        origin, normal, x_dir = axises[aid]

        # Collect boundary points in EDGE_LOOP order (sample curves, not just vertices)
        ordered_pts = []
        EDGE_SAMPLES = 8  # samples per edge for smooth curve approximation
        if plane_id in surface_to_bounds:
            for bound_id in surface_to_bounds[plane_id]:
                if bound_id not in face_outer_bounds: continue
                el_id = face_outer_bounds[bound_id]
                if el_id not in edge_loops: continue
                for oe_id in edge_loops[el_id]:
                    if oe_id not in oriented_edges: continue
                    ec_id, forward = oriented_edges[oe_id]
                    if ec_id not in edge_curves: continue
                    vp1, vp2, cg_id = edge_curves[ec_id]
                    cp1 = vertex_points.get(vp1)
                    cp2 = vertex_points.get(vp2)

                    # Try to sample the B-spline curve for smooth edges
                    if cg_id in bsp_curves and cp1 and cp2 and cp1 in cart_points and cp2 in cart_points:
                        curve = bsp_curves[cg_id]
                        t_min, t_max = curve[3], curve[4]
                        # Sample uniformly (respect orientation)
                        samples = EDGE_SAMPLES
                        for s in range(samples):
                            t_val = t_min + (t_max - t_min) * s / (samples - 1)
                            if not forward:
                                t_val = t_max - (t_val - t_min)
                            pt = eval_curve(curve, t_val)
                            if not ordered_pts or ordered_pts[-1] != pt:
                                ordered_pts.append(pt)
                    elif cp1 and cp2 and cp1 in cart_points and cp2 in cart_points:
                        # Fallback: straight line between vertices
                        if forward:
                            pts = [cart_points[cp1], cart_points[cp2]]
                        else:
                            pts = [cart_points[cp2], cart_points[cp1]]
                        for pt in pts:
                            if not ordered_pts or ordered_pts[-1] != pt:
                                ordered_pts.append(pt)
                break  # only first FACE_OUTER_BOUND (outer loop)

        # Remove duplicate close (last == first)
        if ordered_pts and ordered_pts[-1] == ordered_pts[0]:
            ordered_pts.pop()

        if len(ordered_pts) < 3: continue

        # Fan triangulation from first vertex
        v0 = ordered_pts[0]
        for i in range(1, len(ordered_pts) - 1):
            A, B, C = v0, ordered_pts[i], ordered_pts[i + 1]
            # Bilinear CPs: [A, B; C, C] → triangle ABC
            P = [[A, B], [C, C]]
            # Degree-elevate 1→3 so resolve shader's cubic Bernstein works correctly
            cp4 = [[elev_cp(P, k, l) for l in range(4)] for k in range(4)]
            surfaces.append({
                'p_u': 3, 'p_v': 3,
                'n_u': 3, 'n_v': 3,
                'U': [0,0,0,0, 1,1,1,1],
                'V': [0,0,0,0, 1,1,1,1],
                'cp_grid': cp4,
            })
            plane_tri_count += 1

    if plane_tri_count > 0:
        n_planes = len([1 for m in re.finditer(r'#(\d+)\s*=\s*PLANE\s*\(', text)])
        print(f"  [PLANE] {n_planes} planes -> {plane_tri_count} triangles")

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

    # --- Diagnostic: count all surface types in the STEP file ---
    all_surf_types = {}
    for m in re.finditer(r'#\d+\s*=\s*(\w+)\s*\(', text):
        t = m.group(1)
        if 'SURFACE' in t or 'SURF' in t or t in ('PLANE','CYLINDRICAL','SPHERICAL','CONICAL','TOROIDAL',
            'EXTRUSION','REVOLUTION','SWEPT','BLEND','OFFSET','TRIMMED'):
            all_surf_types[t] = all_surf_types.get(t, 0) + 1
    if all_surf_types:
        bsp_cnt = sum(v for k,v in all_surf_types.items()
                      if k in ('B_SPLINE_SURFACE','B_SPLINE_SURFACE_WITH_KNOTS'))
        plane_cnt = all_surf_types.get('PLANE', 0)
        other = sum(v for k,v in all_surf_types.items()
                    if k not in ('B_SPLINE_SURFACE','B_SPLINE_SURFACE_WITH_KNOTS','PLANE'))
        print(f"STEP surface types found: {dict(sorted(all_surf_types.items()))}")
        parts = [f"{len(surfaces)} total"]
        if bsp_cnt: parts.append(f"{bsp_cnt} B-splines")
        if plane_cnt: parts.append(f"{plane_cnt} planes -> {plane_tri_count if plane_cnt else 0} tris (deg-3 elevated)")
        print(f"Loaded: {', '.join(parts)}")
        if other > 0:
            print(f"WARNING: {other} other surface types SKIPPED")

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
