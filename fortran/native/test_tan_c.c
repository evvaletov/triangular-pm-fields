/* C ABI smoke test for libtan.so. Compile against the Fortran shared library. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void tan_load_c(const char *path, int *status);
void tan_eval_c(const double point[3], int mode, int order, double radius,
                double coeff[3 * 84], int *status);
void tan_batch_c(int count, const double *points, int mode,
                 double *fields, int *statuses);

int main(void)
{
    int status = -1;
    double point[3] = {0.4, 0.3, 0.5};
    double coeff[3 * 84];
    double fields[9];
    double points[9] = {
        0.4, 0.3, 0.5,
        2.0, 2.0, 1.0,
        0.0, 0.0, 0.0
    };
    int statuses[3];
    int i, nfail = 0;
    double bx, by, bz, nrm;

    for (i = 0; i < 3 * 84; ++i) coeff[i] = 0.0;
    tan_load_c("results/triangle.cache", &status);
    if (status != 0) {
        fprintf(stderr, "tan_load_c status %d\n", status);
        return 1;
    }
    tan_eval_c(point, 1, 0, 1.0, coeff, &status);
    if (status != 0) {
        fprintf(stderr, "tan_eval_c status %d\n", status);
        return 1;
    }
    /* Fortran coeff(3,84) is column-major: component varies fastest. */
    bx = coeff[0];
    by = coeff[1];
    bz = coeff[2];
    nrm = sqrt(bx * bx + by * by + bz * bz);
    printf("C scalar Bx By Bz %.16e %.16e %.16e |B|=%.16e\n", bx, by, bz, nrm);
    if (fabs(bx + 5.2789733300147188e-3) > 1e-12) nfail++;
    if (fabs(by + 8.0652631704507933e-4) > 1e-12) nfail++;
    if (fabs(bz - 8.6807780553898276e-2) > 1e-12) nfail++;

    tan_eval_c(point, 1, 7, 1.0, coeff, &status);
    if (status != 5) {
        fprintf(stderr, "order>6 should be status 5, got %d\n", status);
        nfail++;
    }

    tan_batch_c(3, points, 1, fields, statuses);
    if (statuses[0] != 0 || statuses[1] != 0 || statuses[2] != 8) {
        fprintf(stderr, "batch statuses %d %d %d\n",
                statuses[0], statuses[1], statuses[2]);
        nfail++;
    }
    if (fabs(fields[0] - bx) > 0.0 || fabs(fields[1] - by) > 0.0 ||
        fabs(fields[2] - bz) > 0.0) {
        fprintf(stderr, "batch mismatch\n");
        nfail++;
    }
    printf("C ABI nfail=%d\n", nfail);
    return nfail == 0 ? 0 : 1;
}
