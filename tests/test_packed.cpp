// Console regression harness: load a packed .surfb end-to-end and report the
// per-surface sums. The expected line for data/STP/Army Guy.step is
//   nSurf=3506 skipped=0 totalBezierVec4=919744 badFloats=0
// (919744 vec4 x 16 B = 14.03 MB cache, exactly the text-path result — the
// bit-identity canary between packed and text loads). Exit code 0 = match.
// Build: cmake --build build --target test_packed
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../src/nurbs.h"

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: test_packed <file.surfb>\n"); return 2; }
    float** cps = NULL; int *nu = NULL, *nv = NULL, *ru = NULL, *rv = NULL;
    int n = nurbs_read_packed(argv[1], &cps, &nu, &nv, &ru, &rv);
    if (n < 0) { printf("READ FAILED\n"); return 1; }
    long long totalV4 = 0, bad = 0, skipped = 0;
    for (int i = 0; i < n; i++) {
        if (!cps[i]) { skipped++; continue; }
        long long v4 = (long long)nu[i] * nv[i] * ru[i] * rv[i];
        totalV4 += v4;
        for (long long k = 0; k < v4 * 4; k++)
            if (!isfinite(cps[i][k])) bad++;
        free(cps[i]);
    }
    free(cps); free(nu); free(nv); free(ru); free(rv);
    printf("nSurf=%d skipped=%lld totalBezierVec4=%lld badFloats=%lld\n",
           n, skipped, totalV4, bad);
    return (n == 3506 && skipped == 0 && totalV4 == 919744 && bad == 0) ? 0 : 1;
}
