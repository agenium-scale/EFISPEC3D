import sys
import os
import argparse

## -----------------------------------------------------------------------------

src_fr = '''\
\\input{{slides_template}}

\\begin{{document}}

\\scalefirstframe

\\scaletitleframe{{Bibliothèque NSIMD}}
{{Application au calcul d'éléments finis spectraux 3D}}

\\scalesectionframe{{NSIMD}}{{Présentation}}

\\begin{{scaleframe}}{{NSIMD}}{{Généralités}}
\\centering
Bibliothèque de calcul C/C++ \\textaccent{{\\textit{{open source}}}} offrant une 
une abstraction sur les jeux d'instructions SIMD des processeurs.
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
\\end{{scaleframe}}

\\begin{{scaleframe}}{{NSIMD}}{{Motivations}}
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
\\end{{scaleframe}}

\\begin{{scaleframe}}{{NSIMD}}{{Principes}}
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
\\textaccent{{recompilé}} pour l'architecture cible, et non 
\\textaccent{{réécrit}}.
\\end{{scaleframe}}

\\scalesectionframe{{Application}}{{Calcul d'éléments finis spectraux 3D}}


{results}


\\end{{document}}

'''

## -----------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tmp-dir', type=str, required=True,
                        help='Tmp dir location')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    tmp_dir = args.tmp_dir
    out_dir = os.path.join(tmp_dir, 'tex')
    out_file = os.path.join(out_dir, 'slides_efisepc3d.tex')
    src = src_fr
    with open(out_file, 'w') as fp:
        fp.write(src.format(results=''))
