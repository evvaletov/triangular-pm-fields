#include "triangle_pm/native.hpp"
#include "triangle_pm/triangle.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <atomic>
#include <thread>

using triangle_pm::evaluate;
using triangle_pm::evaluate_tokens;
using triangle_pm::kMaxOrder;
using triangle_pm::monomials;
using triangle_pm::nmono;
using triangle_pm::Result;
using triangle_pm::Status;
using triangle_pm::to_json;

namespace {

int npass = 0;
int nfail = 0;

void expect(bool cond, const char* msg) {
  if (cond) {
    ++npass;
    std::cout << "ok " << msg << '\n';
  } else {
    ++nfail;
    std::cerr << "FAIL " << msg << '\n';
  }
}

double hypot3(double x, double y, double z) { return std::sqrt(x * x + y * y + z * z); }

double vnorm(const std::array<double, 3>& a) { return hypot3(a[0], a[1], a[2]); }

double l2(const std::array<double, 3>& a, const std::array<double, 3>& b) {
  return hypot3(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
}

bool scalar_budget(const std::array<double, 3>& got, const std::array<double, 3>& ref) {
  return l2(got, ref) <= 1e-12 + 1e-10 * vnorm(ref);
}

bool coeff_budget(const Result& got, const std::vector<std::array<double, 3>>& ref) {
  double max_ref = 0.0;
  double max_diff = 0.0;
  const int n = static_cast<int>(got.exponents.size());
  if (static_cast<int>(ref.size()) != n) return false;
  for (int k = 0; k < n; ++k) {
    for (int c = 0; c < 3; ++c) {
      max_ref = std::max(max_ref, std::abs(ref[static_cast<std::size_t>(k)][c]));
      max_diff = std::max(
          max_diff, std::abs(got.coefficients[c][static_cast<std::size_t>(k)] -
                             ref[static_cast<std::size_t>(k)][c]));
    }
  }
  return max_diff / std::max(1.0, max_ref) <= 1e-10;
}

void gauss_legendre(int n, std::vector<double>& x, std::vector<double>& w) {
  x.resize(static_cast<std::size_t>(n));
  w.resize(static_cast<std::size_t>(n));
  const double pi = std::acos(-1.0);
  for (int i = 0; i < n; ++i) {
    double z = std::cos(pi * (i + 0.75) / (n + 0.5));
    double dp = 0.0;
    for (int it = 0; it < 80; ++it) {
      double p1 = 1.0, p2 = 0.0;
      for (int j = 0; j < n; ++j) {
        const double p3 = p2;
        p2 = p1;
        p1 = ((2.0 * j + 1.0) * z * p2 - j * p3) / (j + 1.0);
      }
      dp = n * (z * p1 - p2) / (z * z - 1.0);
      const double z1 = z;
      z = z1 - p1 / dp;
      if (std::abs(z - z1) < 1e-16) break;
    }
    x[static_cast<std::size_t>(i)] = z;
    w[static_cast<std::size_t>(i)] = 2.0 / ((1.0 - z * z) * dp * dp);
  }
}

struct Quad {
  std::vector<std::array<double, 3>> nodes;
  std::vector<double> weights;
};

Quad triangle_quadrature(const std::array<double, 3>& a, const std::array<double, 3>& b,
                         const std::array<double, 3>& c, int order) {
  std::vector<double> t, wt;
  gauss_legendre(order, t, wt);
  for (int i = 0; i < order; ++i) {
    t[static_cast<std::size_t>(i)] = 0.5 * (t[static_cast<std::size_t>(i)] + 1.0);
    wt[static_cast<std::size_t>(i)] *= 0.5;
  }
  std::array<double, 3> e1{b[0] - a[0], b[1] - a[1], b[2] - a[2]};
  std::array<double, 3> e2{c[0] - a[0], c[1] - a[1], c[2] - a[2]};
  const double area2 = hypot3(e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2],
                              e1[0] * e2[1] - e1[1] * e2[0]);
  Quad q;
  q.nodes.reserve(static_cast<std::size_t>(order * order));
  q.weights.reserve(static_cast<std::size_t>(order * order));
  for (int i = 0; i < order; ++i) {
    for (int j = 0; j < order; ++j) {
      const double u = t[static_cast<std::size_t>(i)];
      const double v = t[static_cast<std::size_t>(j)];
      const double s = 1.0 - u;
      std::array<double, 3> node;
      for (int k = 0; k < 3; ++k) node[k] = a[k] + u * e1[k] + s * v * e2[k];
      const double w = wt[static_cast<std::size_t>(i)] * wt[static_cast<std::size_t>(j)] * s * area2;
      q.nodes.push_back(node);
      q.weights.push_back(w);
    }
  }
  return q;
}

std::array<double, 3> green_scalar(const Quad& q, const std::array<double, 3>& p,
                                   double sigma) {
  const double fac = sigma / (4.0 * std::acos(-1.0));
  std::array<double, 3> acc{0, 0, 0};
  for (std::size_t i = 0; i < q.nodes.size(); ++i) {
    const double dx = p[0] - q.nodes[i][0];
    const double dy = p[1] - q.nodes[i][1];
    const double dz = p[2] - q.nodes[i][2];
    const double r = hypot3(dx, dy, dz);
    const double w = q.weights[i] * fac / (r * r * r);
    acc[0] += w * dx;
    acc[1] += w * dy;
    acc[2] += w * dz;
  }
  return acc;
}

std::vector<std::array<double, 3>> green_jet(const Quad& q, const std::array<double, 3>& p,
                                             double sigma, int order, double radius) {
  const auto keys = monomials(order);
  const int n = static_cast<int>(keys.size());
  std::vector<int> lookup(7 * 7 * 7, -1);
  for (int i = 0; i < n; ++i) {
    lookup[keys[i][0] * 49 + keys[i][1] * 7 + keys[i][2]] = i;
  }
  const int nnode = static_cast<int>(q.nodes.size());
  std::vector<double> d(static_cast<std::size_t>(3 * nnode));
  std::vector<double> r2(static_cast<std::size_t>(nnode));
  for (int i = 0; i < nnode; ++i) {
    d[static_cast<std::size_t>(3 * i + 0)] = p[0] - q.nodes[i][0];
    d[static_cast<std::size_t>(3 * i + 1)] = p[1] - q.nodes[i][1];
    d[static_cast<std::size_t>(3 * i + 2)] = p[2] - q.nodes[i][2];
    r2[i] = d[static_cast<std::size_t>(3 * i)] * d[static_cast<std::size_t>(3 * i)] +
            d[static_cast<std::size_t>(3 * i + 1)] * d[static_cast<std::size_t>(3 * i + 1)] +
            d[static_cast<std::size_t>(3 * i + 2)] * d[static_cast<std::size_t>(3 * i + 2)];
  }
  std::vector<double> power(static_cast<std::size_t>(n * nnode), 0.0);
  std::vector<double> series(static_cast<std::size_t>(n * nnode), 0.0);
  for (int i = 0; i < nnode; ++i) {
    power[static_cast<std::size_t>(i)] = 1.0;
    series[static_cast<std::size_t>(i)] = 1.0;
  }
  struct Qterm {
    std::array<int, 3> key;
    std::vector<double> values;
  };
  std::vector<Qterm> qs;
  for (int axis = 0; axis < 3; ++axis) {
    Qterm lin;
    lin.key = {0, 0, 0};
    lin.key[axis] = 1;
    lin.values.resize(static_cast<std::size_t>(nnode));
    Qterm quad;
    quad.key = {0, 0, 0};
    quad.key[axis] = 2;
    quad.values.resize(static_cast<std::size_t>(nnode));
    for (int i = 0; i < nnode; ++i) {
      lin.values[i] = 2.0 * d[static_cast<std::size_t>(3 * i + axis)] / r2[i];
      quad.values[i] = 1.0 / r2[i];
    }
    qs.push_back(std::move(lin));
    qs.push_back(std::move(quad));
  }
  double coefficient = 1.0;
  for (int k = 1; k <= order; ++k) {
    std::vector<double> nxt(static_cast<std::size_t>(n * nnode), 0.0);
    for (int ia = 0; ia < n; ++ia) {
      for (const auto& term : qs) {
        const int tx = keys[ia][0] + term.key[0];
        const int ty = keys[ia][1] + term.key[1];
        const int tz = keys[ia][2] + term.key[2];
        if (tx + ty + tz > order) continue;
        const int it = lookup[tx * 49 + ty * 7 + tz];
        if (it < 0) continue;
        for (int i = 0; i < nnode; ++i) {
          nxt[static_cast<std::size_t>(it * nnode + i)] +=
              power[static_cast<std::size_t>(ia * nnode + i)] * term.values[i];
        }
      }
    }
    power.swap(nxt);
    coefficient *= (-1.5 - k + 1.0) / k;
    for (std::size_t i = 0; i < series.size(); ++i) series[i] += coefficient * power[i];
  }
  const double fac = sigma / (4.0 * std::acos(-1.0));
  std::vector<std::array<double, 3>> out(static_cast<std::size_t>(n), {0.0, 0.0, 0.0});
  for (int axis = 0; axis < 3; ++axis) {
    for (int ik = 0; ik < n; ++ik) {
      double acc = 0.0;
      for (int i = 0; i < nnode; ++i) {
        double c = series[static_cast<std::size_t>(ik * nnode + i)] *
                   d[static_cast<std::size_t>(3 * i + axis)];
        if (keys[ik][axis] > 0) {
          std::array<int, 3> prev = keys[ik];
          prev[axis] -= 1;
          const int ip = lookup[prev[0] * 49 + prev[1] * 7 + prev[2]];
          if (ip >= 0) c += series[static_cast<std::size_t>(ip * nnode + i)];
        }
        const double base = q.weights[i] / std::pow(r2[i], 1.5);
        acc += c * base;
      }
      const int deg = keys[ik][0] + keys[ik][1] + keys[ik][2];
      out[static_cast<std::size_t>(ik)][axis] = fac * acc * std::pow(radius, deg);
    }
  }
  return out;
}

std::array<double, 3> eval_poly(const Result& r, const std::array<double, 3>& u) {
  std::array<double, 3> acc{0, 0, 0};
  for (std::size_t i = 0; i < r.exponents.size(); ++i) {
    const double m = std::pow(u[0], r.exponents[i][0]) * std::pow(u[1], r.exponents[i][1]) *
                     std::pow(u[2], r.exponents[i][2]);
    acc[0] += r.coefficients[0][i] * m;
    acc[1] += r.coefficients[1][i] * m;
    acc[2] += r.coefficients[2][i] * m;
  }
  return acc;
}

void test_exponents() {
  expect(nmono(6) == 84, "nmono(6)=84");
  expect(nmono(0) == 1, "nmono(0)=1");
  expect(nmono(1) == 4, "nmono(1)=4");
  expect(nmono(2) == 10, "nmono(2)=10");
  expect(nmono(7) == 0 && nmono(-1) == 0, "invalid nmono");
  const auto ex = monomials(6);
  expect(static_cast<int>(ex.size()) == 84, "84 exponent rows");
  int k = 0;
  bool walk = true;
  for (int q = 0; q <= 6; ++q) {
    for (int a = 0; a <= q; ++a) {
      for (int b = 0; b <= q - a; ++b) {
        const int z = q - a - b;
        if (ex[static_cast<std::size_t>(k)][0] != a ||
            ex[static_cast<std::size_t>(k)][1] != b ||
            ex[static_cast<std::size_t>(k)][2] != z)
          walk = false;
        ++k;
      }
    }
  }
  expect(walk && k == 84, "graded x-then-y exponent walk");
  expect(ex[0][0] == 0 && ex[0][1] == 0 && ex[0][2] == 0, "(0,0,0) first");
  expect(ex[1][0] == 0 && ex[1][1] == 0 && ex[1][2] == 1, "deg1 z");
  expect(ex[2][0] == 0 && ex[2][1] == 1 && ex[2][2] == 0, "deg1 y");
  expect(ex[3][0] == 1 && ex[3][1] == 0 && ex[3][2] == 0, "deg1 x");
  expect(ex[83][0] == 6 && ex[83][1] == 0 && ex[83][2] == 0, "x^6 last");
}

void test_inputs() {
  const std::array<double, 3> a{0, 0, 0}, b{1.2, 0, 0}, c{0.2, 0.9, 0}, p{0.4, 0.3, 0.5};
  expect(evaluate(7, 1, 0.8, a, b, c, p).status == Status::kInvalidOrder, "q>6");
  expect(evaluate(-1, 1, 0.8, a, b, c, p).status == Status::kInvalidOrder, "q<0");
  expect(evaluate(0, 0, 0.8, a, b, c, p).status == Status::kBadRadius, "zero radius");
  expect(evaluate(0, -1, 0.8, a, b, c, p).status == Status::kBadRadius, "negative radius");
  const double inf = std::numeric_limits<double>::infinity();
  const double nan = std::numeric_limits<double>::quiet_NaN();
  expect(evaluate(0, inf, 0.8, a, b, c, p).status == Status::kBadRadius, "inf radius");
  expect(evaluate(0, 1, 0.8, a, b, c, {nan, 0, 0}).status == Status::kNonfiniteObserver,
         "NaN observer");
  expect(evaluate(0, 1, 0.8, a, b, c, {inf, 0, 0}).status == Status::kNonfiniteObserver,
         "Inf observer");
  expect(evaluate(0, 1, nan, a, b, c, p).status == Status::kInvalidGeometry, "NaN charge");
  expect(evaluate(0, 1, 0.8, a, a, c, p).status == Status::kInvalidGeometry, "zero area");
  expect(evaluate(0, 1, 0.8, {inf, 0, 0}, b, c, p).status == Status::kInvalidGeometry,
         "nonfinite vertex");
  expect(evaluate_tokens({"not-a-number"}).status == Status::kInvalidGeometry, "malformed tokens");
  expect(evaluate_tokens({}).status == Status::kInvalidGeometry, "empty tokens");
}

void test_zero_charge() {
  const std::array<double, 3> a{0, 0, 0}, b{1.2, 0, 0}, c{0.2, 0.9, 0};
  auto z = evaluate(2, 0.1, 0.0, a, b, c, {0.4, 0.3, 0.5});
  expect(z.status == 0, "zero charge regular");
  bool all0 = true;
  for (int i = 0; i < 3; ++i)
    for (double v : z.coefficients[i])
      if (v != 0.0) all0 = false;
  expect(all0 && static_cast<int>(z.exponents.size()) == 10, "zero charge all slots");
  auto edge = evaluate(0, 1, 0.0, a, b, c, {0.6, 0, 0});
  expect(edge.status == 0 && edge.coefficients[0][0] == 0.0, "zero charge on edge");
  auto vert = evaluate(3, 0.2, 0.0, a, b, c, a);
  expect(vert.status == 0, "zero charge on vertex");
  auto sheet = evaluate(4, 0.05, 0.0, a, b, c, {0.4, 0.3, 0.0});
  expect(sheet.status == 0, "zero charge on interior");
}

void test_boundaries() {
  const std::array<double, 3> a{0, 0, 0}, b{1.2, 0, 0}, c{0.2, 0.9, 0};
  expect(evaluate(0, 1, 0.8, a, b, c, a).status == Status::kPhysicalEdgeVertex, "vertex A");
  expect(evaluate(0, 1, 0.8, {0.1, 0, 0}, {1.2, 0, 0}, {0.2, 0.9, 0}, {0.1, 0, 0}).status ==
             Status::kPhysicalEdgeVertex,
         "binary64 0.1 vertex via exact rational round-trip");
  expect(evaluate(0, 1, 0.8, a, b, c, b).status == Status::kPhysicalEdgeVertex, "vertex B");
  expect(evaluate(1, 0.1, 0.8, a, b, c, {0.6, 0, 0}).status == Status::kPhysicalEdgeVertex,
         "edge midpoint DA");
  expect(evaluate(0, 1, 0.8, a, b, c, {0.6, 0, 0}).status == Status::kPhysicalEdgeVertex,
         "edge midpoint scalar");
  expect(evaluate(2, 0.01, 0.8, a, b, c, {0.4, 0.3, 0.0}).status == Status::kDaOnChargedSheet,
         "interior DA rejected");
  auto avg = evaluate(0, 1, 0.8, a, b, c, {0.4, 0.3, 0.0});
  expect(avg.status == 0, "interior scalar average accepted");
  expect(std::abs(avg.coefficients[2][0]) <= 1e-14, "Bz=0 on xy interior");
  auto ext = evaluate(2, 0.02, 0.8, a, b, c, {2.0, 2.0, 0.0});
  expect(ext.status == 0, "exterior-plane DA accepted");
  expect(std::abs(ext.coefficients[2][0]) <= 1e-14, "exterior Bz const 0");
  expect(std::abs(ext.coefficients[2][1]) > 1e-8, "exterior dBz/du_z nonzero");
}

void test_scalar_quadrature() {
  const std::array<double, 3> a{0, 0, 0}, b{1.2, 0, 0}, c{0.2, 0.9, 0};
  const double sigma = 0.8;
  const std::array<double, 3> p{0.4, 0.3, 0.5};
  const Quad q32 = triangle_quadrature(a, b, c, 32);
  const Quad q48 = triangle_quadrature(a, b, c, 48);
  const auto r32 = green_scalar(q32, p, sigma);
  const auto r48 = green_scalar(q48, p, sigma);
  expect(l2(r32, r48) <= 1e-11, "quadrature refinement 32 vs 48");
  auto got = evaluate(0, 1, sigma, a, b, c, p);
  expect(got.status == 0, "regular scalar status");
  const std::array<double, 3> g{got.coefficients[0][0], got.coefficients[1][0],
                                got.coefficients[2][0]};
  expect(scalar_budget(g, r48), "regular scalar vs Green quadrature");

  const std::array<double, 3> pfar{2.0, 2.0, 1.0};
  auto far = evaluate(0, 1, sigma, a, b, c, pfar);
  const auto rfar = green_scalar(q48, pfar, sigma);
  const std::array<double, 3> gf{far.coefficients[0][0], far.coefficients[1][0],
                                 far.coefficients[2][0]};
  expect(far.status == 0 && scalar_budget(gf, rfar), "far scalar vs quadrature");

  const std::array<double, 3> pm{0.4, 0.3, -0.5};
  auto gm = evaluate(0, 1, sigma, a, b, c, pm);
  const auto rm = green_scalar(q48, pm, sigma);
  const std::array<double, 3> gmv{gm.coefficients[0][0], gm.coefficients[1][0],
                                  gm.coefficients[2][0]};
  expect(gm.status == 0 && scalar_budget(gmv, rm), "below-plane scalar vs quadrature");
}

void test_jets_and_radius() {
  const std::array<double, 3> a{0, 0, 0}, b{1.2, 0, 0}, c{0.2, 0.9, 0};
  const double sigma = 0.8;
  const std::array<double, 3> p{0.4, 0.3, 0.5};
  const Quad q = triangle_quadrature(a, b, c, 40);
  for (int order : {1, 2, 4, 6}) {
    auto got = evaluate(order, 1.0, sigma, a, b, c, p);
    expect(got.status == 0, "jet status");
    expect(static_cast<int>(got.exponents.size()) == nmono(order), "all monomials present");
    const auto ref = green_jet(q, p, sigma, order, 1.0);
    expect(coeff_budget(got, ref), "jet vs Green kernel expansion");
  }
  auto r1 = evaluate(3, 1.0, sigma, a, b, c, p);
  auto r2 = evaluate(3, 0.25, sigma, a, b, c, p);
  bool scale_ok = r1.status == 0 && r2.status == 0;
  for (std::size_t i = 0; i < r1.exponents.size(); ++i) {
    const int deg = r1.exponents[i][0] + r1.exponents[i][1] + r1.exponents[i][2];
    const double f = std::pow(0.25, deg);
    for (int c = 0; c < 3; ++c) {
      const double g = r2.coefficients[c][i];
      const double e = r1.coefficients[c][i] * f;
      const double den = std::max(1.0, std::abs(e));
      if (std::abs(g - e) / den > 1e-10) scale_ok = false;
    }
  }
  expect(scale_ok, "radius rescaling of every coefficient");

  auto jet = evaluate(6, 0.1, sigma, a, b, c, p);
  const std::array<double, 3> u{0.02, -0.01, 0.008};
  const auto pred = eval_poly(jet, u);
  std::array<double, 3> qpt{p[0] + 0.1 * u[0], p[1] + 0.1 * u[1], p[2] + 0.1 * u[2]};
  auto sc = evaluate(0, 1, sigma, a, b, c, qpt);
  const std::array<double, 3> truth{sc.coefficients[0][0], sc.coefficients[1][0],
                                    sc.coefficients[2][0]};
  expect(scalar_budget(pred, truth), "Taylor reconstruction (factorial + radius)");
}

void test_transforms() {
  const std::array<double, 3> a{0, 0, 0}, b{1.2, 0, 0}, c{0.2, 0.9, 0};
  const std::array<double, 3> p{0.4, 0.3, 0.5};
  const double sigma = 0.8;
  auto abc = evaluate(2, 0.1, sigma, a, b, c, p);
  auto acb = evaluate(2, 0.1, sigma, a, c, b, p);
  bool rev = abc.status == 0 && acb.status == 0;
  for (int k = 0; k < nmono(2); ++k)
    for (int c = 0; c < 3; ++c)
      if (std::abs(abc.coefficients[c][k] - acb.coefficients[c][k]) >
          1e-10 * std::max(1.0, std::abs(abc.coefficients[c][k])))
        rev = false;
  expect(rev, "reversed winding, fixed sigma");

  const std::array<double, 3> d{1.0, -2.0, 0.3};
  auto tr = evaluate(1, 0.2, sigma, {a[0] + d[0], a[1] + d[1], a[2] + d[2]},
                     {b[0] + d[0], b[1] + d[1], b[2] + d[2]},
                     {c[0] + d[0], c[1] + d[1], c[2] + d[2]},
                     {p[0] + d[0], p[1] + d[1], p[2] + d[2]});
  auto base = evaluate(1, 0.2, sigma, a, b, c, p);
  bool trans = tr.status == 0 && base.status == 0;
  for (int k = 0; k < nmono(1); ++k)
    for (int c = 0; c < 3; ++c)
      if (std::abs(tr.coefficients[c][k] - base.coefficients[c][k]) >
          1e-10 * std::max(1.0, std::abs(base.coefficients[c][k])))
        trans = false;
  expect(trans, "translation invariance");

  const double s = 2.0;
  auto scaled = evaluate(2, 0.2, sigma, {a[0] * s, a[1] * s, a[2] * s},
                         {b[0] * s, b[1] * s, b[2] * s}, {c[0] * s, c[1] * s, c[2] * s},
                         {p[0] * s, p[1] * s, p[2] * s});
  auto unscaled = evaluate(2, 0.1, sigma, a, b, c, p);
  bool scale_inv = scaled.status == 0 && unscaled.status == 0;
  for (int k = 0; k < nmono(2); ++k)
    for (int c = 0; c < 3; ++c)
      if (std::abs(scaled.coefficients[c][k] - unscaled.coefficients[c][k]) >
          1e-10 * std::max(1.0, std::abs(unscaled.coefficients[c][k])))
        scale_inv = false;
  expect(scale_inv, "homologous length scaling of vertices, point, radius");

  const std::array<double, 3> oa{0, 0, 0}, ob{2, -1, 0}, oc{3, 0, -1};
  const std::array<double, 3> n{1, 2, 3};
  const double nn = vnorm(n);
  std::array<double, 3> op{1.25, -0.25, -0.25};
  for (int k = 0; k < 3; ++k) op[k] += 0.4 * n[k] / nn;
  const Quad oq = triangle_quadrature(oa, ob, oc, 40);
  auto og = evaluate(0, 1, 1.0, oa, ob, oc, op);
  const auto oref = green_scalar(oq, op, 1.0);
  const std::array<double, 3> ogv{og.coefficients[0][0], og.coefficients[1][0],
                                  og.coefficients[2][0]};
  expect(og.status == 0 && scalar_budget(ogv, oref), "oblique n=(1,2,3) vs quadrature");
  auto osheet = evaluate(0, 1, 1.0, oa, ob, oc, {1.25, -0.25, -0.25});
  expect(osheet.status == 0, "oblique exact-sheet scalar");
  expect(evaluate(2, 0.01, 1.0, oa, ob, oc, {1.25, -0.25, -0.25}).status ==
             Status::kDaOnChargedSheet,
         "oblique exact-sheet DA rejected");
}

void test_json_and_cli_tokens() {
  const std::vector<std::string> tok = {"0",  "1",  "0.8", "0", "0", "0", "1.2", "0",
                                        "0",  "0.2", "0.9", "0", "0.4", "0.3", "0.5"};
  auto r = evaluate_tokens(tok);
  expect(r.status == 0, "15-token CLI parse");
  const std::string js = to_json(r);
  expect(js.find("\"status\":0") != std::string::npos, "JSON status 0");
  expect(js.find("nan") == std::string::npos && js.find("inf") == std::string::npos &&
             js.find("Inf") == std::string::npos,
         "JSON finite");
  auto bad = evaluate_tokens({"7", "1", "0.8", "0", "0", "0", "1", "0", "0", "0", "1", "0", "0",
                              "0", "1"});
  expect(bad.status == Status::kInvalidOrder, "token q=7");
  const std::string js_bad = to_json(bad);
  expect(js_bad.find("\"exponents\":[]") != std::string::npos &&
             js_bad.find("\"coefficients\":[]") != std::string::npos,
         "error JSON empty arrays");
}

void test_native_optional() {
  expect(triangle_pm::native_eval_tokens({"missing", "cache", "0", "0", "1", "1e100", "0", "1"}).status == 4, "native oversized mode rejected before integer cast");
  expect(triangle_pm::native_eval_tokens({"missing", "cache", "0", "0", "1", "1", "1e100", "1"}).status == Status::kInvalidOrder, "native oversized order rejected before integer cast");
  const char* lib = std::getenv("TRIANGLE_NATIVE_LIB");
  const char* cache = std::getenv("TRIANGLE_NATIVE_CACHE");
  if (!lib || !cache) {
    std::cout << "ok native skipped (set TRIANGLE_NATIVE_LIB and TRIANGLE_NATIVE_CACHE)\n";
    ++npass;
    return;
  }
  auto r = triangle_pm::native_eval(lib, cache, {0.4, 0.3, 0.5}, 1, 0, 1.0);
  expect(r.status == 0, "native scalar status");
  if (r.status == 0) {
    expect(static_cast<int>(r.exponents.size()) == 1, "native nmono 1");
    expect(std::isfinite(r.coefficients[0][0]) && std::isfinite(r.coefficients[1][0]) &&
               std::isfinite(r.coefficients[2][0]),
           "native finite components");
  }
  auto bad = triangle_pm::native_eval(lib, cache, {0.4, 0.3, 0.5}, 1, 7, 1.0);
  expect(bad.status == Status::kInvalidOrder, "native q>6 status 5");
}

}  // namespace

int main() {
  // Simultaneous first use exercises the one-time shared table initialization.
  std::atomic<int> ready{0};
  std::atomic<bool> start{false};
  std::array<Result, 8> parallel;
  std::vector<std::thread> workers;
  for (int i=0; i<8; ++i) workers.emplace_back([&, i] {
    ++ready;
    while (!start.load()) std::this_thread::yield();
    parallel[i] = evaluate(2, .125, 1.25, {0.,0.,0.}, {1.,0.,0.},
                           {0.,1.,0.}, {.25,.25,.75});
  });
  while (ready.load()!=8) std::this_thread::yield();
  start = true;
  for (auto& worker : workers) worker.join();
  for (const auto& result : parallel)
    expect(result.status==0 && result.coefficients==parallel[0].coefficients,
           "concurrent standalone first use");
  test_exponents();
  test_inputs();
  test_zero_charge();
  test_boundaries();
  test_scalar_quadrature();
  test_jets_and_radius();
  test_transforms();
  test_json_and_cli_tokens();
  test_native_optional();
  std::cout << "PASS " << npass << " FAIL " << nfail << '\n';
  return nfail == 0 ? 0 : 1;
}
