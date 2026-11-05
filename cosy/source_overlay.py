"""Fail-closed overlay on build_cosy.sh; the production script stays intact."""
import os
from pathlib import Path
import sys


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one build anchor: {old!r}")
    return text.replace(old, new, 1)


def prepare(target):
    root = Path(os.environ["TRIANGLE_ROOT"]).resolve()
    output = Path(os.environ["TRIANGLE_OUTPUT"]).resolve()
    forbidden = {(root / name).resolve() for name in ('cosy', 'cosy_custom', 'COSY.bin')}
    if output in forbidden or output.parent == root or output.name.lower() in {'cosy', 'cosy_custom', 'cosy.bin'}:
        raise RuntimeError("TRIANGLE_OUTPUT must be an isolated target, not a production COSY executable")
    text = (root / "cosy_ext/build_cosy.sh").read_text()
    start = text.index('SCRIPT_DIR="$(cd ')
    stop = text.index('EXT_DIR="${PROJECT_ROOT}/cosy_ext"')
    text = text[:start] + '''PROJECT_ROOT="${TRIANGLE_ROOT}"
PRISTINE="${TRIANGLE_PRISTINE:-${PROJECT_ROOT}/pristine/COSY 10.2/UNIX}"
if [[ ! -d "${PRISTINE}" && -z "${TRIANGLE_PRISTINE:-}" ]]; then
    PRISTINE="${PROJECT_ROOT}/../cosy-pristine/COSY 10.2/UNIX"
fi
''' + text[stop:]
    text = replace_once(text, '# Copy pristine source', '''# Native assembly additions participate in every COMMON capacity patch.
cp "${TRIANGLE_BRIDGE_SOURCE}" "${TMPDIR}/"
cp "${TRIANGLE_SOURCE}" "${TMPDIR}/tan_mod.f90"
if [[ -f "$(dirname "${TRIANGLE_SOURCE}")/tan_jet.f90" ]]; then
    cp "$(dirname "${TRIANGLE_SOURCE}")/tan_jet.f90" "${TMPDIR}/"
fi
cp "${TRIANGLE_EXACT_SOURCE}" "${TMPDIR}/"
# Copy pristine source''')
    text = replace_once(text, 'LVAR_FILES="${FOXY} dafox.f',
                        'LVAR_FILES="triangle_assembly_proc.f ${FOXY} dafox.f')
    text = replace_once(text, 'for DA_FILE in daini_wrapper.f',
                        'for DA_FILE in triangle_assembly_proc.f daini_wrapper.f')
    anchor = '# ── Fixed-form margin check on the PATCHED tree'
    text = replace_once(text, anchor, '''python3 "${TRIANGLE_DISPATCH_SCRIPT}" dispatch foxy.f
TRIANGLE_JET_OBJ=""
if [[ -f tan_jet.f90 ]]; then
    ${FC} ${F90FLAGS} -O3 -ffree-line-length-none -c tan_jet.f90 -o tan_jet.o
    TRIANGLE_JET_OBJ="tan_jet.o"
fi
${FC} ${F90FLAGS} -O3 -ffree-line-length-none -c tan_mod.f90 -o tan_mod.o
gcc -std=c11 -O3 -fPIC -c triangle_exact_geometry.c -o triangle_exact_geometry.o
${FC} ${FFLAGS} -c triangle_assembly_proc.f -o triangle_assembly_proc.o

''' + anchor)
    text = replace_once(text, '    ${GPU_OBJS} ${EMBED_OBJS}',
                        '    triangle_assembly_proc.o tan_mod.o ${TRIANGLE_JET_OBJ} triangle_exact_geometry.o \\\n    ${GPU_OBJS} ${EMBED_OBJS}')
    text = replace_once(text, '${HDF5_LINK} ${PIC_LIBS} 2>&1',
                        '${HDF5_LINK} ${PIC_LIBS} -lgmp 2>&1')
    Path(target).write_text(text)


def dispatch(target):
    path = Path(target)
    text = path.read_text()
    if 'LSUB=248' not in text or 'PRO*1488' not in text:
        raise RuntimeError('Unexpected intrinsic table dimensions')
    text = text.replace('LSUB=248', 'LSUB=249').replace('PRO*1488', 'PRO*1494')
    text = replace_once(text,
        "DATA PRO(1471:1488)  / 'SPCKDASPCHYB$$$$$$' /",
        "DATA PRO(1471:1494)  / 'SPCKDASPCHYBTRLOADTRFLD ' /")
    text = replace_once(text, '392,393,394,395,396,397,398),IGOTO',
                        '392,393,394,395,396,397,398,399),IGOTO')
    blocks = []
    for label, name, nargs in [(398, 'TRLOAD', 2), (399, 'TRFLD', 8)]:
        args = [f'IA {i}' for i in range(1, nargs+1)]
        lines = [f'  {label} CONTINUE', f'      NARG = {nargs}',
                 '      IF(NCOD(IC+4)-IC-5.NE.NARG) GOTO 2900']
        lines += [f'      {a} = NADD(NCOD(IC+{i+6}))'
                  for i, a in enumerate(args)]
        lines += [f'      CALL {name}(' + ','.join(args) + ')',
                  '      IC = NCOD(IC+2)', '      GOTO 1000']
        blocks.extend(lines)
    text = replace_once(text, '  398 GOTO 2000', '\n'.join(blocks))
    path.write_text(text)


if __name__ == '__main__':
    {'prepare': prepare, 'dispatch': dispatch}[sys.argv[1]](sys.argv[2])
