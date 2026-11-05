#!/usr/bin/env julia
# Native client for the Fortran tan_mod physical kernel (shared C ABI).
# This is not the standalone Julia evaluator; it loads libtan.so and a
# TAN_CACHE_V1 cache and calls tan_load_c / tan_eval_c.
# Usage:
#   julia --project=. native_cli.jl LIBRARY CACHE X Y Z MODE ORDER RADIUS
#   julia --project=. native_cli.jl --batch
# The Fortran cache is process-global and is not thread-safe.
include(joinpath(@__DIR__, "src", "TriangularPMFields.jl"))
using .TriangularPMFields

const USAGE = "usage: native_cli.jl LIBRARY CACHE X Y Z MODE ORDER RADIUS"

function emit(r::FieldResult)
    write_result_json(stdout, r)
    println()
    return r.status
end

function run_one(tokens::Vector{String})
    return emit(TriangularPMFields.eval_native_tokens(tokens))
end

function run_batch()
    any_fail = false
    for line in eachline(stdin)
        tokens = split(strip(line))
        isempty(tokens) && continue
        st = run_one(String.(tokens))
        if st != 0
            any_fail = true
        end
    end
    return any_fail ? 1 : 0
end

function main(args)
    if !isempty(args) && (args[1] == "--help" || args[1] == "-h")
        println(USAGE)
        println("Shares the Fortran tan_mod physical kernel via the C ABI.")
        println("Not the standalone Julia field implementation.")
        println("The native cache is process-global; do not share across threads.")
        println("--batch reads whitespace-separated argument lines from stdin.")
        return 0
    end
    if !isempty(args) && args[1] == "--batch"
        length(args) == 1 || return emit(TriangularPMFields.eval_native_tokens(String[])) == 0 ? 0 : 1
        return run_batch()
    end
    st = run_one(String.(args))
    return st == 0 ? 0 : 1
end

exit(main(ARGS))
