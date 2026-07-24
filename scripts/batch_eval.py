#!/usr/bin/env python3
"""Process multiple .surf files into one merged OBJ, one process per surface."""
import sys, os, subprocess, tempfile, glob

def main():
    if len(sys.argv) < 3:
        print("Usage: python batch_eval.py <surf_dir> <output.obj>")
        return

    surf_dir = sys.argv[1]
    out_path = sys.argv[2]
    surfs = sorted(glob.glob(os.path.join(surf_dir, '*.surf')))

    total_surfs, total_verts, offset = 0, 0, 0
    tmp = tempfile.NamedTemporaryFile(suffix='.obj', delete=False, mode='w')
    tmp_name = tmp.name
    tmp.close()

    exe = 'build/nurbs_eval.exe'
    print(f"Processing {len(surfs)} surfaces from {surf_dir}...")

    for idx, sf in enumerate(surfs):
        # Run single-surface eval into temp OBJ
        subprocess.run([exe, sf], capture_output=True, timeout=120)
        src = 'build/nurbs_output.obj'
        if not os.path.exists(src):
            continue

        with open(src) as f:
            lines = f.readlines()

        # Count vertices from this surface
        n_verts = sum(1 for l in lines if l.startswith('v '))

        # Append with offset
        with open(tmp_name, 'a') as out:
            for line in lines:
                if line.startswith('v '):
                    out.write(line)
                elif line.startswith('f '):
                    # Re-index faces with current offset
                    parts = line.split()
                    new_parts = ['f']
                    for p in parts[1:]:
                        new_parts.append(str(int(p) + offset))
                    out.write(' '.join(new_parts) + '\n')

        offset += n_verts
        total_surfs += 1
        total_verts += n_verts
        if (idx + 1) % 50 == 0:
            print(f"  {idx+1}/{len(surfs)}  verts={total_verts}")

    os.rename(tmp_name, out_path)
    print(f"Done: {total_surfs} surfaces, {total_verts} vertices -> {out_path}")


if __name__ == '__main__':
    main()
