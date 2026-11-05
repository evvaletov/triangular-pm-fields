/* C ABI smoke test for libtan.so. Compile against the Fortran shared library. */
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/resource.h>
#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>
#include <signal.h>

void tan_load_c(const char *path, int *status);
void tan_eval_c(const double point[3], int mode, int order, double radius,
                double coeff[3 * 84], int *status);
void tan_batch_c(int count, const double *points, int mode,
                 double *fields, int *statuses);

/* Each malformed load runs in its own bounded process: an unchecked allocation
 * must fail this test even when it terminates the Fortran runtime. */
static int rejected_reload(const char *label, const char *text, long padding,
                           int expected_status)
{
    char path[128];
    snprintf(path, sizeof(path), "results/reload_%s.cache", label);
    FILE *file = fopen(path, "w");
    if (!file) return 1;
    if (fputs(text, file) == EOF) { fclose(file); return 1; }
    if (padding && (fseek(file, padding - 1, SEEK_SET) || fputc('\n', file) == EOF)) {
        fclose(file); return 1;
    }
    if (fclose(file)) return 1;
    pid_t child = fork();
    if (child < 0) return 1;
    if (child == 0) {
        struct rlimit memory, cpu;
        if (getrlimit(RLIMIT_AS, &memory) || getrlimit(RLIMIT_CPU, &cpu)) _exit(10);
        if (memory.rlim_cur > 128UL*1024*1024) memory.rlim_cur = 128UL*1024*1024;
        if (cpu.rlim_cur > 20) cpu.rlim_cur = 20;
        if (setrlimit(RLIMIT_AS, &memory) || setrlimit(RLIMIT_CPU, &cpu)) _exit(11);
        alarm(30);
        int status;
        double point[3] = {.4, .3, .5}, coeff[252];
        tan_load_c("results/triangle.cache", &status);
        if (status != 0) _exit(12);
        tan_eval_c(point, 1, 0, 1., coeff, &status);
        if (status != 0 || !(coeff[2] > 0.)) _exit(13);
        tan_load_c(path, &status);
        if (status != expected_status) _exit(14);
        for (int i = 0; i < 252; ++i) coeff[i] = 123.;
        tan_eval_c(point, 1, 0, 1., coeff, &status);
        if (status != 1) _exit(15);
        for (int i = 0; i < 252; ++i) if (coeff[i] != 0.) _exit(16);
        tan_load_c("results/triangle.cache", &status);
        if (status != 0) _exit(17);
        tan_eval_c(point, 1, 0, 1., coeff, &status);
        if (status != 0 || !(coeff[2] > 0.)) _exit(18);
        _exit(0);
    }
    int outcome;
    if (waitpid(child, &outcome, 0) != child) return 1;
    int failed = !WIFEXITED(outcome) || WEXITSTATUS(outcome) != 0;
    printf("C ABI bounded reload %s: %s (wait=%d, expected status=%d)\n",
           label, failed ? "FAIL" : "PASS", outcome, expected_status);
    return failed;
}

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
    nfail += rejected_reload("huge_planes", "TAN_CACHE_V1\n2147483647 0\n", 0, 3);
    nfail += rejected_reload("huge_edges", "TAN_CACHE_V1\n0 2147483647\n", 0, 3);
    nfail += rejected_reload("huge_both", "TAN_CACHE_V1\n2147483647 2147483647\n", 0, 3);
    nfail += rejected_reload("truncated_header", "TAN_CACHE_V1\n1\n", 0, 3);
    nfail += rejected_reload("truncated_geometry", "TAN_CACHE_V1\n1 0\n", 0, 3);
    /* Sparse padding passes the byte plausibility check. The geometry array
     * alone exceeds RLIMIT_AS, exercising the STAT= allocation-failure path. */
    nfail += rejected_reload("allocation_failure", "TAN_CACHE_V1\n1000000 0\n", 11000000, 2);
    printf("C ABI nfail=%d\n", nfail);
    return nfail == 0 ? 0 : 1;
}
