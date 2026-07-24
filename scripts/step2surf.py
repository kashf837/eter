#!/usr/bin/env python3
"""Convert STEP B_SPLINE_SURFACE to .surf (supports inline and entity formats)."""
import sys, os, re

def parse_step_file(path):
    """Read STEP file, return (entities dict, list of surface data dicts)."""
    with open(path) as f:
        text = f.read()

    # Collect all CARTESIAN_POINT entities
    cart_points = {}
    for m in re.finditer(r'#(\d+)\s*=\s*CARTESIAN_POINT\s*\(\s*\'[^\']*\'\s*,\s*\(\s*([^)]+)\s*\)\s*\)\s*;', text):
        cid = int(m.group(1))
        xyz = [float(v.strip()) for v in m.group(2).split(',')]
        cart_points[cid] = xyz

    surfaces = []

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

                # Parse CP grid rows — each row is (#id, #id, ...)
                rows = re.findall(r'\(\s*((?:#\d+[\s,]*)+)\)', cp_grid_raw)
                cp_grid = []
                for row in rows:
                    ids = [int(s.strip().lstrip('#')) for s in row.split(',') if s.strip()]
                    row_pts = []
                    for cid in ids:
                        if cid in cart_points:
                            row_pts.append(cart_points[cid])
                    if row_pts:
                        cp_grid.append(row_pts)

                if len(cp_grid) < 2:
                    continue

                # Parse knot data: (u_mults), (v_mults), (u_knots), (v_knots)
                lists = re.findall(r'\(([0-9eE.\-+,\s]+)\)', knot_raw)
                if len(lists) >= 4:
                    u_mult  = [int(float(v.strip())) for v in lists[0].split(',') if v.strip()]
                    v_mult  = [int(float(v.strip())) for v in lists[1].split(',') if v.strip()]
                    u_knots = [float(v.strip()) for v in lists[2].split(',') if v.strip()]
                    v_knots = [float(v.strip()) for v in lists[3].split(',') if v.strip()]

                    U = []
                    for kval, km in zip(u_knots, u_mult):
                        U.extend([kval] * km)
                    V = []
                    for kval, km in zip(v_knots, v_mult):
                        V.extend([kval] * km)

                    surf = {
                        'p_u': p_u, 'p_v': p_v,
                        'n_u': len(cp_grid) - 1,
                        'n_v': len(cp_grid[0]) - 1,
                        'U': U, 'V': V,
                        'cp_grid': cp_grid
                    }
                    surfaces.append(surf)
            except Exception as e:
                pass

    return surfaces


def parse_standalone_surface(body, cart_points):
    """Parse Format A body."""
    parts = split_body(body)
    p_u = int(parts[1].strip())
    p_v = int(parts[2].strip())

    rows = re.findall(r'\(([^)]+)\)', parts[3])
    cp_grid = []
    for row in rows:
        ids = [int(s.strip().lstrip('#')) for s in row.split(',') if s.strip().startswith('#')]
        row_pts = [cart_points[cid] for cid in ids if cid in cart_points]
        if row_pts:
            cp_grid.append(row_pts)

    knot_section = ','.join(parts[4:])
    lists = re.findall(r'\(([0-9.,\s]+)\)', knot_section)
    if len(lists) < 4:
        return None
    u_mult  = [int(float(v.strip())) for v in lists[0].split(',') if v.strip()]
    v_mult  = [int(float(v.strip())) for v in lists[1].split(',') if v.strip()]
    u_knots = [float(v.strip()) for v in lists[2].split(',') if v.strip()]
    v_knots = [float(v.strip()) for v in lists[3].split(',') if v.strip()]

    U, V = [], []
    for k, m in zip(u_knots, u_mult): U.extend([k] * m)
    for k, m in zip(v_knots, v_mult): V.extend([k] * m)

    return {'p_u': p_u, 'p_v': p_v, 'n_u': len(cp_grid)-1, 'n_v': len(cp_grid[0])-1,
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
    print(f"Wrote {path}: {s['n_u']+1}x{s['n_v']+1} CPs, order {s['p_u']}x{s['p_v']}")


def main():
    all_flag = '--all' in sys.argv
    args = [a for a in sys.argv[1:] if a != '--all']
    if len(args) < 1:
        print("Usage: python step2surf.py [--all] <input.STEP> [output]")
        return

    infile = args[0]
    surfaces = parse_step_file(infile)

    if not surfaces:
        print("No surfaces found")
        return

    print(f"Found {len(surfaces)} surfaces")

    if all_flag:
        base = args[1] if len(args) > 1 else infile.replace('.STEP', '').replace('.step', '')
        os.makedirs(os.path.dirname(base) or '.', exist_ok=True)
        for idx, s in enumerate(surfaces):
            write_surf(f"{base}_{idx:04d}.surf", s)
    else:
        out = args[1] if len(args) > 1 else infile.replace('.STEP', '.surf').replace('.step', '.surf')
        write_surf(out, surfaces[0])


if __name__ == '__main__':
    main()
