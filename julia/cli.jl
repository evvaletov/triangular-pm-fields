#!/usr/bin/env julia
# Standalone triangle μ0 H CLI. One JSON object per call. No Fortran.
# Usage:
#   julia --project=. cli.jl ORDER RADIUS SIGMA AX AY AZ BX BY BZ CX CY CZ PX PY PZ
#   julia --project=. cli.jl --batch   # one argument line per stdin record
include(joinpath(@__DIR__, "src", "TriangularPMFields.jl"))
using .TriangularPMFields

const USAGE = "usage: cli.jl ORDER RADIUS SIGMA AX AY AZ BX BY BZ CX CY CZ PX PY PZ"

function emit(r::FieldResult)
    write_result_json(stdout, r)
    println()
    return r.status
end

function run_one(tokens::Vector{String})
    return emit(eval_standalone_tokens(tokens))
end

function run_batch()
    any_fail = false
    for line in eachline(stdin)
        tokens = split(strip(line))
        isempty(tokens) && continue
        st = run_one(String.(tokens))
        if st != STATUS_OK
            any_fail = true
        end
    end
    return any_fail ? 1 : 0
end

function main(args)
    if !isempty(args) && (args[1] == "--help" || args[1] == "-h")
        println(USAGE)
        println("Standalone evaluator; does not call the Fortran kernel.")
        println("--batch reads whitespace-separated argument lines from stdin.")
        return 0
    end
    if !isempty(args) && args[1] == "--batch"
        length(args) == 1 || return emit(eval_standalone_tokens(String[])) == 0 ? 0 : 1
        return run_batch()
    end
    st = run_one(String.(args))
    return st == STATUS_OK ? 0 : 1
end

exit(main(ARGS))
