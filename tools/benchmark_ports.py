"""Reproducible standalone CLI batch cost, including startup, JIT, parsing and JSON.

This measures end-to-end calls, not warm library kernels. The C++ and Julia
implementations use different internal precision. Every output is checked
against a frozen independent reference before its timing is retained.
"""
import argparse
import json
from pathlib import Path
import statistics
import time
import numpy as np

from conformance import compare, execute_batch

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build',type=Path,default=ROOT/'build')
    parser.add_argument('--julia',default='julia')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    cases=json.loads((ROOT/'corpus/cases.json').read_text())['cases']
    references={r['id']:r for r in json.loads((ROOT/'corpus/reference.json').read_text())['results']}
    providers={'cpp':[str(args.build.resolve()/'cpp/triangle_cli'),'--batch'],
               'julia':[args.julia,'--startup-file=no','--project='+str(ROOT/'julia'),str(ROOT/'julia/cli.jl'),'--batch']}
    records=[]
    for order in (1,2,6):
        case=next(c for c in cases if c['id']==f'regular_above_q{order}')
        line=[case['order'],case['radius'],case['sigma'],
              *np.asarray(case['vertices']).ravel().tolist(),*case['point']]
        batch=[line]*16
        for provider,command in providers.items():
            samples=[]
            for trial in range(4):
                start=time.perf_counter();results=execute_batch(command,batch)
                elapsed=time.perf_counter()-start
                for result in results:
                    problems,_=compare(case,result,references[case['id']])
                    if problems:raise RuntimeError(str(problems))
                if trial:samples.append(elapsed)
            records.append(dict(provider=provider,order=order,cases_per_process=len(batch),
                process_startup_and_jit_included=True,parsing_and_json_included=True,
                excluded_process_warmups=1,wall_s=samples,median_wall_s=statistics.median(samples)))
    report=dict(schema_version=1,scope=__doc__,results=records)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(records,indent=2))


if __name__=='__main__':main()
