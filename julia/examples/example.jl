#!/usr/bin/env julia
# Regular-point μ0 H for a right triangle. Standalone evaluator only.
include(joinpath(@__DIR__, "..", "src", "TriangularPMFields.jl"))
using .TriangularPMFields

order = 2.0
radius = 0.1
sigma = 0.8
A = (0.0, 0.0, 0.0)
B = (2.0, 0.0, 0.0)
C = (0.0, 2.0, 0.0)
P = (0.4, 0.3, 0.5)

r = triangle_map(order, radius, sigma, A..., B..., C..., P...)
println("standalone status ", r.status)
println("nmono ", length(r.exponents))
if r.status == STATUS_OK
    println("Bx By Bz (constant) ",
            r.coefficients[1, 1], ' ',
            r.coefficients[2, 1], ' ',
            r.coefficients[3, 1])
    u = (0.2, -0.1, 0.05)
    rec = evaluate_map(r.coefficients, r.exponents, u)
    println("map at u ", rec)
end
write_result_json(stdout, r)
println()
