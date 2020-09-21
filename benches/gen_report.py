# Use utf-8 encoding
# -*- coding: utf-8 -*-

import sys
import os
import argparse

## -----------------------------------------------------------------------------

doc = '''\
\\input{{report_template}}

\\begin{{document}}

{intro}

{benches}

\\end{{document}}
'''

intro_fr = '''\
\\scaletitle{Résultats des benches pour NSIMD sur EFISPEC 3D}

\\section{À propos de ce document}
Ce document présente les résultats de l'exécution d'un noyau de calcul pour la
méthode des éléments finis spectraux 3D (EFISPEC3D) sur différentes 
architectures matérielles. Pour chacun des cas, nous avons utilisé trois 
versions différentes de l'algorithme : une version scalaire de référence, une 
version vectorisée à l'aide du jeu SIMD nati de l'architecture cible, et une
version utilisant notre bibliothèque de calcul NSIMD.

\\section{La bibliothèque NSIMD}
Agenium Scale développe et maintient depuis 2019 la bibliothèque Open Source
NSIMD, dont la première version est disponible sur
Github\\footnote{https://github.com/agenium-scale/nsimd}. Il s'agit d'une
bibliothèque C et C++ permettant d'écrire du code SIMD indépendamment d'une
architecture particulière. Un code écrit à l'aide de NSIMD peut ainsi s'exécuter
sur toutes les architectures supportées par la bibliothèque, ce qui permet de
diminuer les coûts de développement liés à la vectorisation d'un algorithme.

NSIMD propose des opérateurs SIMD génériques qui sont, si possible, de simples
wrappers vers les opérateurs correspondants sur l'architecture cible. C'est le
cas pour les opérations arithmétiques de base, comme l'addition, ou la
multiplication par exemple. Dans le cas contraire, les opérateurs sont émulés
sur l'architecture, en utilisant de préférence des opérations entre registres
SIMD.

NSIMD n'est pas la seule bibliothèque proposant d'abstraire les opérations
vectorielles, mais, à notre
connaissance, elle est la seule à proposer un support pour le jeu d'instructions
SVE dont les premiers pocesseurs seront bientôt disponibles. NSIMD permet
également d'écrire du code SIMD et gère de manière transparente les tailles de
registres. Un code écrit à l'aide de NSIMD permet de générer à la fois des
exécutables optimisés sur une architecture dont la taille des registres est
fixe, ou sur une architecture (comme SVE), où la taille de ces registres n'est
connue qu'à l'exécution.

\\subsection{Fonctionnement}
La bibliothèque NSIMD repose sur les capacités des compilateurs à
\\textit{inliner} les fonctions (c'est-à-dire, remplacer leurs appels dans le
programme par le code correspondant) lors de l'utilisation de l'option de
compilation adéquate (en général, {\\tt -O2} et au-delà). De cette manière, un
appel à une fonction de NSIMD est directement remplacé par le code ou
l'opérateur SIMD correspondant. Cela permet d'abstraire du code SIMD sans coût
supplémentaire. Les principaux compilateurs, comme GCC, Clang, ICC et MSVC,
proposent cette passe d'optimisation. Afin de permettre ces optimisations, la
plupart du code source de NSIMD est placée dans des fichiers d'en-tête, seules
les plus grosses fonctions sont placées dans des fichiers sources à part, et
compilées sous forme d'une bibliothèque dynamique.

\\subsection{Jeux d'instructions supportés par NSIMD}
Actuellement, NSIMD support les jeux d'instructions suivants :
\\begin{itemize}
\\item[$\\bullet$] Intel :
  \\begin{itemize}
  \\item[-] SSE2
  \\item[-] SSE4.2
  \\item[-] AVX
  \\item[-] AVX2
  \\item[-] AVX512 (support pour KNL et Xeon Skylake)
  \\end{itemize}
\\item[$\\bullet$] Arm :
  \\begin{itemize}
  \\item[-] NEON 128 bits (ARmv7 et Aarch64)
  \\item[-] SVE
  \\end{itemize}
\\end{itemize}

\\subsection{Types supportés}
NSIMD supporte tous les types entiers signés et non signés existant sur $8$ à
$64$ bits. Le support des nombres à virgule flottante se fait pour $16$, $32$,
et $64$ bits. Lorsqu'une architecture ne supporte pas nativement les nombres à
virgule flottante sur $16$ bits, leur fontionnement est émulé. NSIMD est
également, à notre connaissance, la seule bibliothèque SIMD à proposer un
support pour les nombres à virgule flottante sur $16$ bits.

\\subsection{Interfaces de programmation proposées}
NSIMD est compatible avec les standards C89, C++98, C++11 et C++14. Elle propose
trois interfaces de
programmation\footnote{https://agenium-scale.github.io/nsimd/index.html}. La
première est une interface C, et propose des opérateurs de la forme :
\\texttt{nsimd\\_\\{op\\}\\_ \\{ext\}\\_\\{type\\}([args])}. Où \\texttt{op} 
est le nom de l'opérateur utilisé, \\texttt{ext} représente l'extension SIMD 
utilisée, et \\texttt{type} représente le type utilisé, il peut être de la forme :
\\texttt{\\{i, u, f\\}\\{8, 16, 32, 64\\}}.  Ce code est rendu générique par la
définition de macros permettant d'écrire un code de la forme:
\\texttt{v\{op\}([args], \{type\})}. La définition de l'architecture cible à la
compilation permet d'appeler les bonnes fonctions.

Les deux autres interfaces de programmation sont des interfaces C++. La première
interface fournit des opérateurs génériques de la forme :
\\texttt{nsimd::\\{op\\}([args], type())}, le type à utiliser étant défini en
paramètre. Enfin, la deuxième interface définit un type générique
\\texttt{nsimd::pack <\\{type\\}, \\{N\\}>} représentant un registre SIMD ainsi qu'un
niveau de déroulage {\\tt N} (par défaut {\\tt 1}), permettant un déroulage
automatique des boucles. Les opérateurs sont sous la forme générique :
\\texttt{nsimd::{op}<\\{type\\}>([args])}.

\\section{Application au calcul d'éléments finis spectraux}
\\subsection{EFISPEC3D}
EFISPEC3D\\footnote{http://efispec.free.fr/} est une bibliothèque logicielle de 
simulation sismique utilisant la méthode de éléments spectraux finis en 3D.

Cette bibliothèque est très répandue pour la simulation sismique, et l'optimisation
de ses performances est un enjeu important pour accélérer la rapidité des 
simulations. Actuellement, les optimisations de cet algorithme reposent sur les
capacités des processeurs à auto-vecotriser du code. Cependant, les gains en
performances ne sont pas aussi bons qu'en vectorisant explicitement le code.
Nous comparons ici les performances du cde de calcul d'EFISPEC3D vectorisées
explicitement pour une architecture particulière, et une version équivalente
vectorisée à l'aide de NSIMD.

\\input{figures/efispec}

'''

benches_fr = '''\
\\section{{Résultats}}

Cette section détaille les résultats de l'évaluation de la boucle de calcul 
principale de calcul des éléments spectraux finis en 3D pour chacun des 
architectures considérées. Dans chaque cas, nous avons exécuté trois versions
différentes de ce code de calcul : une version scalaire, une version optimisée
avec le jeu d'instructions disponible pour l'architecture considérée, et une 
dernière version optimisée avec NSIMD.
L'objectif étant bien entendu de se rapprocher, en utilisant NSIMD,  des 
performances obtenues en optimisant le code avec les instructions natives.

Dans chacun des cas, nous avons établi une un graphe montrant le temps mis 
pour exécuter la boucle principale pour effectuer le calcul sur un même jeu de 
données.
\\newpage
{results}
'''

## -----------------------------------------------------------------------------

comp_version_template = '''\
\\subsection{{Compiler version}}
\\begin{{lstlisting}}[frame=single, style=info]
{}
\\end{{lstlisting}}
'''
def get_compiler_version(root, lang='fr'):
    with open(os.path.join(root, 'compiler.info'), 'r') as fp:
        comp_info = fp.read()
    return comp_version_template.format(comp_info)    

os_descr_template = '''\
\\subsection{{Système d'exploitation}}
\\begin{{lstlisting}}[frame=single, style=info]
{}
\\end{{lstlisting}}
'''
def get_os_description(root, lanf='fr'):
    with open(os.path.join(root, 'uname.info'), 'r') as fp:
        os_info = fp.read()
    return os_descr_template.format(os_info)

cpu_info_template = '''\
\\subsection{{Informations sur l'architecture CPU}}
\\begin{{lstlisting}}[frame=single, style=info]
{}
\\end{{lstlisting}}
'''
def get_cpu_info(root, lang='fr'):
    with open(os.path.join(root, 'cpu.info'), 'r') as fp:
        proc_info = fp.read()
    return cpu_info_template.format(proc_info)

mem_info_template = '''\
\\subsection{{Informations sur la RAM}}
\\begin{{lstlisting}}[frame=single, style=info]
{}
\\end{{lstlisting}}
'''
def get_mem_info(root, lang='fr'):
    with open(os.path.join(root, 'memory.info'), 'r') as fp:
        mem_info = fp.read()
    return mem_info_template.format(mem_info)

libc_info_template = '''\
\\subsection{{Information sur la bibliothèque standard}}
\\begin{{lstlisting}}[frame=single, style=info]
{}
\\end{{lstlisting}}
'''
def get_libc_info(root, lang='fr'):
    with open(os.path.join(root, 'libc.info'), 'r') as fp:
        libc_info = fp.read()
    return libc_info_template.format(libc_info)

def gen_intro(root, lang='fr'):
    ret = intro_fr
    ret += get_compiler_version(root, lang)
    ret += get_os_description(root, lang)
    ret += get_cpu_info(root, lang)
    ret += get_mem_info(root, lang)
    ret += get_libc_info(root, lang)
    return ret

def gen_benches_list(graph_list, lang='fr'):
    results = '\n'.join('\\input{{graphs/{}}}'.format(f.split('.')[0]) \
                        for f in graph_list)
    return benches_fr.format(results=results)

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
    out_file = os.path.join(out_dir, 'benches_efispec3d.tex')
    graphs_dir = os.path.join(tmp_dir, 'tex/graphs')
    graphs = os.listdir(graphs_dir)

    intro = gen_intro(tmp_dir, 'en')
    benches = gen_benches_list(graphs, 'en')

    src = doc.format(intro=intro, benches=benches)
    with open(out_file, 'w') as fp:
        fp.write(src)
    
