# Copyright (C) 2021  Agenium Scale
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

import matplotlib.pyplot as plt
import numpy as np
import os
import sys

# -----------------------------------------------------------------------------

DEBUG = False

def debug(msg):
    if DEBUG:
        print('DEBUG: {}'.format(msg))

# -----------------------------------------------------------------------------

def draw_graph(output_filename, title, data):
    simd_exts_in_order = ["CPU", "SSE2", "NSIMD SSE2", "SSE42", "NSIMD SSE42",
                          "AVX", "NSIMD AVX", "AVX2", "NSIMD AVX2",
                          "AVX512 KNL", "NSIMD AVX512 KNL", "AVX512 SKYLAKE",
                          "NSIMD AVX512 SKYLAKE", "NEON128", "NSIMD NEON128",
                          "AARCH64", "NSIMD AARCH64"]

    #def gen_svg(title, xlabel, mapping, svg):
    plt.rcParams.update({'figure.autolayout': True})
    plt.style.use('ggplot')
    fig, ax = plt.subplots()

    labels = []
    values = []
    for simd_ext in simd_exts_in_order:
        if not simd_ext in data:
            continue
        labels.append(simd_ext)
        values.append(data[simd_ext])
    values = tuple(values)
    labels = tuple(labels)
    n = len(labels)

    debug(title)
    debug('data   = {}'.format(data))
    debug('values = {}'.format(values))
    debug('labels = {}'.format(labels))
    debug('- - - - -')

    ind = np.arange(n)
    bars = ax.barh(ind, values)
    for i in range(n):
        if labels[i].find('NSIMD') == -1:
            bars[i].set_color('sandybrown')
        else:
            bars[i].set_color('cadetblue')
        ax.text(0, i, ' {:0.2f}'.format(values[i]), color='black', va='center')
    if title != 'ns/day':
        ax.set(title=title, xlabel='milliseconds')
    else:
        ax.set(title=title)
    plt.gca().invert_yaxis()
    plt.yticks(ind, labels)
    plt.savefig(output_filename, transparent=True)

# -----------------------------------------------------------------------------

def avg_of_median(v):
    v.sort()
    i0 = len(v) // 4
    i1 = max(i0, 3 * len(v) // 4)
    return sum(v[i0:i1]) / (i1 - i0 + 1)

# -----------------------------------------------------------------------------

if __name__ == "__main__":
    res = dict()
    for filename in sys.argv[1:]:
        debug('input = {}'.format(filename))
        if not os.path.isfile(filename):
            print('-- Warning: no performance file named ' + filename)
            continue
        ext = ' '.join((filename.split('/')[-1]).split('-')[0:-1]).upper()
        with open(filename) as fin:
            v = []
            for line in fin:
                if line.strip() != '':
                    v.append(float(line.strip()))
            res[ext] = int(avg_of_median(v))
    draw_graph('timings.pdf', 'Timings', res)

