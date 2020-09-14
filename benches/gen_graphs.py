import sys
import os
import argparse

## -----------------------------------------------------------------------------

template = '''\
\\subsection{{Benches results for {simd_ext}}}
\\begin{{figure}}[!ht]
\\centering
\\begin{{tikzpicture}}
\\begin{{axis}}[
    ybar, ymin=0, bar width=1.5cm,
    ylabel={{Execution time(ms)}},
    symbolic x coords={{scalar, {simd_ext}, nsimd-{simd_ext}}},
    xtick=data]
\\addplot coordinates {{
        (scalar, {val0}) 
        ({simd_ext}, {val1}) 
        (nsimd-{simd_ext}, {val2})}};
\\end{{axis}}
\\end{{tikzpicture}}
\\end{{figure}}
'''

## -----------------------------------------------------------------------------

def read_value(filename):
    with open(filename, 'r') as fp:
        val = fp.read().rstrip('\n')
    return val

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ext', type=str, required=True,
                        help='SIMD extension')
    parser.add_argument('--tmp-dir', type=str, required=True,
                        help='Tmp dir')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    simd_ext = args.ext
    root = args.tmp_dir
    out_dir = os.path.join(root, 'tex/graphs')
    out_file = os.path.join(out_dir, simd_ext + '.tex')
    
    scalar = os.path.join(root, 'scalar.log')
    simd = os.path.join(root, simd_ext + '.log')
    nsimd = os.path.join(root, 'nsimd-' + simd_ext + '.log')

    val0 = read_value(scalar)
    val1 = read_value(simd)
    val2 = read_value(nsimd)

    with open(out_file, 'w') as fp:
        fp.write(template.format(simd_ext=simd_ext,
                                 val0=val0, val1=val1, val2=val2))
    
