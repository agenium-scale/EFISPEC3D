# Use utf-8 encoding
# -*- coding: utf-8 -*-

import sys
import os
import argparse

## -----------------------------------------------------------------------------

src_fr = '''\
\\documentclass[shrink, compress, mathserif, 10pt, xcolor=dvipsnames,
  aspectratio=169]{{beamer}}
\\usetheme{{Scale}}

\\usepackage{{textcomp}}
\\usepackage[french]{{babel}}
\\usepackage[T1]{{fontenc}}
\\usepackage[utf8]{{inputenc}}
\\usepackage{{helvet}}
\\usepackage{{avant}}

\\title{{Bibliothèque NSIMD}}
\\subtitle{{Application au calcul d'éléments spectraux finis 3D}}
\\date{{\\today}}
\\author{{Agenium Scale}}

\\begin{{document}}

\\begin{{frame}}[plain]
  \\maketitle
\\end{{frame}}

\\begin{{frame}}{{NSIMD}}{{Présentation}}
\\centering
Bibliothèque de calcul C/C++ \\textcolor{{primary}}{{\\textit{{open source}}}} 
offrant une abstraction sur les jeux d'instructions SIMD des processeurs.
\\newline
NSIMD offre une abstraction sur les jeux d'instructions SIMD propres à chaque 
architecture et permet de réduire les coûts de développement pour l'optimisation
d'un code de calcul.

\\begin{{itemize}}
  \\item Support de différents jeux d'instructions : SSE2, SSE4.2, AVX, AVX2,
  AVX512, Arm NEON, SVE.
  \\item Compatible avec C89, C++98, C++11.
  \\item Abstraction sans coût supplémentaire.
\\end{{itemize}}
\\end{{frame}}

\\begin{{frame}}{{NSIMD}}{{Motivations}}
Nécessité de vectoriser du code à la main :
\\begin{{itemize}}
  \\item Jeux d'instructions multiples
  \\item Hétérogénéités même au sein d'une même famille de processeurs
  \\item Limitation de compilateurs à auto-vectoriser du code.
\\end{{itemize}}
L'optimisation d'un code de calcul passe par l'écriture d'un code spécifique 
pour chaque famille de processeur viblée.
\nesline
Besoin de réécrire ce code pour chaque nouveau jeu d'instructions à supporter.
\\end{{frame}}

\\begin{{frame}}{{NSIMD}}{{Principes}}
\\centering
Utilisation des capacités des compilateurs à inliner des fonctions lors de la 
phase d'optimisation.
\\begin{{itemize}}
  \\item Abstraction des opérateurs SIMD sans coût en termes de performances.
\\end{{itemize}}

\\begin{{itemize}}
  \\item Wrappers autour des instructions existantes si possible
  \\item Émulation des opérateurs non disponibles nativement.
\\end{{itemize}}

Un programme écrit à l'aide de NSIMD n'a besoin que d'être 
\\textcolor{{primary}}{{recompilé}} pour l'architecture cible, et non 
\\textcolor{{primary}}{{réécrit}}.
\\end{{frame}}

\\begin{{frame}}{{Application}}{{Calcul d'éléments finis spectraux 3D}}
\\centering
EFISPEC3D\\footnote{{http://efispec.free.fr/}} est une bibliothèque logicielle de 
simulation sismique utilisant la méthode de éléments spectraux finis en 3D.
\\hfill
\\newline
\\includegraphics[width=.3\\textwidth]{{img/efispec1.png}}
\\includegraphics[width=.3\\textwidth]{{img/efispec2.png}}

\\begin{{itemize}}
  \\item Très utilisée pour la simulation numérique
  \\item Nécessité d'optimiser le code manuellement
\\end{{itemize}}
\\end{{frame}}

\\begin{{frame}}{{Application}}{{Utilisation de NSIMD sur EFISPEC3D}}
Utilisation de NSIMD pour la vectorisation du noyau de calcul de l'algorithme
EFISPEC3D.
\\newline
Comparaison de NSIMD par rapport aux instructions natives pour :
\\begin{{itemize}}
  \\item NEON128
  \\item AACH64
\\end{{itemize}}
Tests compilés avec {comp} sur un processeur {arch}.
\\end{{frame}}

{benches}

\\end{{document}}

'''

## -----------------------------------------------------------------------------

results_template = '''\
\\begin{{frame}}{{Résultats}}
{results}
\\end{{frame}}
'''
def gen_benches_list(graph_list, lang='fr'):
		ret = ''
		for f in graph_list:
			results = '\\input{{graphs/{}}}'.format(f.split('.')[0])
			ret += results_template.format(results=results)
		return ret

## -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp-dir', type=str, required=True,
                        help='Tmp dir location')
    parser.add_argument('--comp', type=str, required=True)
    parser.add_argument('--arch', type=str, required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    comp = args.comp
    arch = args.arch
    tmp_dir = args.tmp_dir
    out_dir = os.path.join(tmp_dir, 'tex')
    out_file = os.path.join(out_dir, 'slides_efispec3d.tex')
    graphs_dir = os.path.join(tmp_dir, 'tex/graphs')
    graphs = os.listdir(graphs_dir)
    src = src_fr

    benches = gen_benches_list(graphs, 'en')
    
    with open(out_file, 'w') as fp:
        fp.write(src.format(benches=benches, comp=comp, arch=arch))
