#!/usr/bin/python

import numpy as np
import matplotlib
from os import getenv
if not getenv('DISPLAY'):
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse

# By default, expects list of tuples of Y values.  With -x, the first
# element of each tuple is instead interpreted as the X value for the
# remaining Y values.
def plot_line(lines):
	if args.xcoord:
		if len(lines[0]) < 2:
			sys.stderr.write("'line -x' requires multiple columns")
			exit(1)
		xs = [float(l[0]) for l in lines]
		cols = range(1, len(lines[0]))
	else:
		xs = range(0, len(lines))
		cols = range(0, len(lines[0]))

	for i, c in enumerate(cols):
		ys = [float(l[c]) for l in lines]
		label = args.legend[i] if args.legend is not None else None

		kw = {}
		for a in ["colors", "markers", "linestyles"]:
			arg = getattr(args, a)
			if arg is not None:
				v = arg[i % len(arg)]
				# HACK.
				kw[a[:-1]] = v

		plt.plot(xs, ys, label=label, **kw)

	if args.legend is not None:
		plt.legend(loc=0)

# expects list of single values
def plot_hist(lines):
	vals = [float(x[0]) for x in lines]
	plt.hist(vals, args.nbins, normed=args.norm)

# expects list of (x, y) tuples
def plot_scatter(lines):
	xs = [float(l[0]) for l in lines]
	ys = [float(l[1]) for l in lines]
	plt.scatter(xs, ys, marker='x')

# expects list of (label, value) pairs
def plot_bars(lines):
	labels = [l[0] for l in lines]
	values = [[float(x) for x in l[1:]] for l in lines]
	grpsize = len(values[0])
	ngrps = len(values)
	w = 0.8

	for i in range(0, grpsize):
		color = args.colors[i % len(args.colors)] if args.colors else None
		base = i*w/grpsize
		try:
			label = args.legend[i]
		except:
			label = None

		try:
			plt.bar(np.arange(base, ngrps+base), [v[i] for v in values],
			        width=w/grpsize, color=color, label=label)
		except ValueError:
			sys.stderr.write("Bar plot error, perhaps '%s' is an invalid"
			                 " color?\n" % color)
			exit(1)

	plt.xticks([i + w/2 for i in range(0, len(lines))], labels,
	           rotation=args.labelangle, rotation_mode="anchor",
	           ha=["center", "right", "left"][cmp(args.labelangle, 0.0)])
	plt.xlim(w-1, len(lines))

	if args.legend is not None:
		plt.legend(loc=0)

# expects a list of single values
def plot_cdf(lines):
	vals = [float(x[0]) for x in lines]
	plt.hist(vals, args.nbins, cumulative=True, histtype='step', normed=args.norm)
	plt.xlim(0, max(vals))
	plt.ylim(0, 1 if args.norm else len(vals))

# expects list of (label, start, end) tuples
def plot_timechart(lines):
	ids = sorted(set([l[0] for l in lines]))
	idnums = dict([(j, i) for (i, j) in enumerate(ids)])

	for label in ids:
		parts = [l for l in lines if l[0] == label]
		base = idnums[label]
		ydim = (base, 1)
		xdims = []
		for p in parts:
			start = float(p[1])
			if args.duration:
				diff = float(p[2])
			else:
				end = float(p[2])
				diff = end-start
			xdims.append((start, diff))
		plt.broken_barh(xdims, ydim, alpha=0.5)

	plt.yticks([x + 0.5 for x in xrange(0, len(idnums))], ids)

def main():
	lines = sys.stdin.readlines()
	if ',' in lines[0]:
		lines = [l.split(',') for l in lines]
	else:
		lines = [l.split() for l in lines]

	for a in ["legend", "colors", "markers", "linestyles"]:
		if hasattr(args, a):
			v = getattr(args, a)
			setattr(args, a, v.split(',') if v is not None else None)

	args.plotmode(lines)

	if args.xlabel:
		plt.xlabel(args.xlabel)
	if args.ylabel:
		plt.ylabel(args.ylabel)
	if args.title:
		plt.title(args.title)

	if args.ylim is not None:
		ylo, yhi = [float(x) for x in args.ylim.split(',')]
		plt.ylim(ylo, yhi)

	xgeom, ygeom = [float(s) for s in args.geometry.split(',')]
	plt.gcf().set_size_inches(xgeom, ygeom, forward=True)

	if args.outfile:
		plt.savefig(args.outfile, dpi=args.dpi)
	else:
		plt.show()

if __name__ == "__main__":
	mainparser = argparse.ArgumentParser(description="plot data from stdin")

	subparsers = mainparser.add_subparsers()

	# python 2.7 doesn't support aliases in add_parser, sadly.
	lineparser = subparsers.add_parser("line", help="draw line plot")
	lineparser.set_defaults(plotmode=plot_line)
	lineparser.add_argument('-x', "--xcoord", action="store_const", const=True,
	                        default=False, help="use first column as X coordinates")
	lineparser.add_argument('-m', "--markers", type=str, help="marker styles")
	lineparser.add_argument('-s', "--linestyles", type=str, help="line styles")

	scatterparser = subparsers.add_parser("scatter", help="draw scatter plot")
	scatterparser.set_defaults(plotmode=plot_scatter)

	histparser = subparsers.add_parser("hist", help="draw histogram")
	histparser.set_defaults(plotmode=plot_hist)

	barparser = subparsers.add_parser("bar", help="draw bar chart")
	barparser.set_defaults(plotmode=plot_bars)
	barparser.add_argument('-r', "--angle", dest="labelangle", type=float,
	                       help="label angle (degrees)", default=0.0)

	for p in [lineparser, barparser]:
		p.add_argument('-L', "--legend", type=str, help="labels for legend")
		p.add_argument('-c', "--colors", type=str,
		               help="color(s) of plotted bars/lines, comma-separated")

	cdfparser = subparsers.add_parser("cdf", help="draw cumulative distribution")
	cdfparser.set_defaults(plotmode=plot_cdf)

	timechartparser = subparsers.add_parser("tc", help="draw timechart")
	timechartparser.set_defaults(plotmode=plot_timechart)

	for p in [histparser, cdfparser]:
		p.add_argument('-b', "--bins", dest="nbins", type=int, metavar="NBINS",
		               default=15, help="number of bins (default 15)")
		p.add_argument('-a', "--absolute", dest="norm", action="store_const",
		               const=False, default=True, help="Don't normalize y-axis")

	timechartparser.add_argument('-d', "--duration", action="store_const",
	                             const=True, default=False, help="read data "
	                             "items as (label, start, length) instead of "
	                             "default (label, start, end)")

	mainargs = [(('-t', "--title"), dict(type=str, help="plot title")),
	            (('-x', "--xlabel"), dict(type=str, help="x-axis label")),
	            (('-y', "--ylabel"), dict(type=str, help="y-axis label")),
	            (('-Y', "--ylim"), dict(type=str, help="y-axis bounds")),
	            (('-o', "--outfile"),
	             dict(type=str, help="file to save plot in (default none)")),
	            (('-r', "--dpi"),
	             dict(type=int, help="resolution of output file (dots per inch)")),
	            (('-g', "--geometry"),
	             dict(type=str, default="8,6",
	                  help="figure geometry in X,Y format (inches)")),
	            (('-l', "--live"),
	             dict(action="store_const", const=True, default=False,
	                  help="update plot as data appears"))]

	for args, kwargs in mainargs:
		mainparser.add_argument(*args, **kwargs)

	args = mainparser.parse_args()

	if not args.outfile and not getenv('DISPLAY'):
		sys.stderr.write("No output file specified but DISPLAY not set\n")
		exit(1)

	main()
