import sys
import os
import argparse

## -----------------------------------------------------------------------------

doc = '''\
\\input{{report}}
\\usepackage{{pgfplots}}
\\pgfplotsset{{width=7cm,compat=1.8}}
\\title{{\\includegraphics[width=27em]{{scale.png}}}}

\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}

\\begin{{document}}

{intro}

{benches}

\\end{{document}}
'''

intro_fr = '''\
\section{Description}
Ceci est un test.
'''

intro_en = '''\
\\maketitle
\\clearpage
\\pagenumbering{gobble}

\\section{Introduction}
\\subsection{About this document}
This document presents the results of the execution of a 3D finite spectral 
elements kernel over different architectures. This document shows the execution
times of the kernel with different setup, including SIMD using native 
instructions like (Arm NEON) compared to a scalar version, and the same version
using the NSIMD library.
'''

benches_en = '''\
\\section{{Benches results}}
{results}
'''

## -----------------------------------------------------------------------------

comp_version_template = '''\
\\subsection{{Compiler version}}
\\begin{{lstlisting}}[frame=single]
{}
\\end{{lstlisting}}
'''
def get_compiler_version(root, lang='fr'):
    with open(os.path.join(root, 'compiler.info'), 'r') as fp:
        comp_info = fp.read()
    return comp_version_template.format(comp_info)    

os_descr_template = '''\
\\subsection{{Operating system}}
\\begin{{lstlisting}}[frame=single]
{}
\\end{{lstlisting}}
'''
def get_os_description(root, lanf='fr'):
    with open(os.path.join(root, 'uname.info'), 'r') as fp:
        os_info = fp.read()
    return os_descr_template.format(os_info)

cpu_info_template = '''\
\\subsection{{Information about the CPU architecture}}
\\begin{{lstlisting}}[frame=single]
{}
\\end{{lstlisting}}
'''
def get_cpu_info(root, lang='fr'):
    with open(os.path.join(root, 'cpu.info'), 'r') as fp:
        proc_info = fp.read()
    return cpu_info_template.format(proc_info)

mem_info_template = '''\
\\subsection{{Information about the RAM}}
\\begin{{lstlisting}}[frame=single]
{}
\\end{{lstlisting}}
'''
def get_mem_info(root, lang='fr'):
    with open(os.path.join(root, 'memory.info'), 'r') as fp:
        mem_info = fp.read()
    return mem_info_template.format(mem_info)

libc_info_template = '''\
\\subsection{{Information about the standard library}}
\\begin{{lstlisting}}[frame=single]
{}
\\end{{lstlisting}}
'''
def get_libc_info(root, lang='fr'):
    with open(os.path.join(root, 'libc.info'), 'r') as fp:
        libc_info = fp.read()
    return libc_info_template.format(libc_info)

def gen_intro(root, lang='fr'):
    ret = intro_en
    ret += get_compiler_version(root, lang)
    ret += get_os_description(root, lang)
    ret += get_cpu_info(root, lang)
    ret += get_mem_info(root, lang)
    ret += get_libc_info(root, lang)
    return ret

def gen_benches_list(graph_list, lang='fr'):
    results = '\n'.join('\\newpage\n\\input{{graphs/{}}}'.format(f.split('.')[0]) \
                        for f in graph_list)
    return benches_en.format(results=results)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp-dir', type=str, required=True,
                        help='Tmp dir location')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tmp_dir = args.tmp_dir
    out_dir = os.path.join(tmp_dir, 'tex')
    out_file = os.path.join(out_dir, 'benches_efispec3d.tex')
    graphs_dir = os.path.join(tmp_dir, 'tex/graphs')
    graphs = os.listdir(graphs_dir)

    intro = gen_intro(tmp_dir, 'en')
    benches = gen_benches_list(graphs, 'en')

    src = doc.format(intro=intro, benches=benches)
    with open(out_file, 'w') as fp:
        fp.write(src)
    
