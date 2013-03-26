#!/usr/bin/python

import numpy as np
import matplotlib
from os import getenv
if not getenv('DISPLAY'):
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import argparse

cmap = plt.cm.get_cmap()

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
		for a in ["markers", "linestyles"]:
			arg = getattr(args, a)
			if arg is not None:
				v = arg[i % len(arg)]
				# HACK.
				kw[a[:-1]] = v

		color = cmap(float(i) / float(max(len(cols), 2)-1))
		plt.plot(xs, ys, label=label, color=color, **kw)

	if args.legend is not None:
		plt.legend(loc=0)

def percentile(vals, pct):
	num = min(int((pct/100.0) * len(vals)), len(vals))
	return sorted(vals)[:num]

# expects list of single values
def plot_hist(lines):
	vals = percentile([float(x[0]) for x in lines], args.percentile)
	if args.range is not None:
		rlo, rhi = [float(x) for x in args.range.split(',')]
		r = rlo, rhi
	else:
		r = None
	plt.hist(vals, args.nbins, range=r, normed=args.norm, color=cmap(0.5))

# expects list of (x, y) tuples
def plot_scatter(lines):
	xs = [float(l[0]) for l in lines]
	ys = [float(l[1]) for l in lines]
	plt.scatter(xs, ys, marker='x')

# expects list of (label, value) pairs
def plot_bars(lines):
	labels = [l[0] for l in lines]
	values = [[float(x) for x in l[1:]] for l in lines]
	ncols = len(values[0])
	nrows = len(values)
	w = 0.8

	prevset = [0 for v in values]
	for i in range(0, ncols):
		if args.stack:
			lefts = np.arange(0, nrows)
		else:
			base = i*w/ncols
			lefts = np.arange(base, nrows+base)

		try:
			label = args.legend[i]
		except:
			label = None

		color = cmap(float(i) / float((max(ncols, 2)-1)))
		thisset = [v[i] for v in values]
		width = w if args.stack else w/ncols
		plt.bar(lefts, thisset, bottom=prevset, width=width, label=label,
		        color=color)

		if args.stack:
			prevset = [x+y for x,y in zip(thisset, prevset)]

	plt.xticks([i + w/2 for i in range(0, len(lines))], labels,
	           rotation=args.labelangle, rotation_mode="anchor",
	           ha=["center", "right", "left"][cmp(args.labelangle, 0.0)])
	plt.xlim(w-1, len(lines))

	if args.legend is not None:
		plt.legend(loc=0)

# expects a list of single values
def plot_cdf(lines):
	vals = [float(x[0]) for x in lines]
	if args.range is not None:
		rlo, rhi = [float(x) for x in args.range.split(',')]
		r = rlo, rhi
	else:
		r = None
	plt.hist(vals, args.nbins, range=r, cumulative=True, histtype='step',
	         normed=args.norm)
	if args.log:
		plt.xscale('log')
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
	global cmap
	lines = [l for l in sys.stdin.readlines() if l[0] != '#']
	if ',' in lines[0]:
		lines = [l.split(',') for l in lines]
	else:
		lines = [l.split() for l in lines]

	for a in ["legend", "markers", "linestyles"]:
		if hasattr(args, a):
			v = getattr(args, a)
			setattr(args, a, v.split(',') if v is not None else None)

	if args.colormap is not None:
		cmap = plt.cm.get_cmap(args.colormap)
		if cmap is None:
			sys.stderr.write("invalid colormap: %s\n" % args.colormap)
			sys.stderr.write("Available colormaps:\n\t")
			mapnames = (c for c in plt.cm.datad.keys())
			forward = (n for n in mapnames if not n.endswith("_r"))
			sys.stderr.write("\n\t".join(sorted(forward, key=str.lower)))
			sys.stderr.write('\n')
			exit(1)

	args.plotmode(lines)

	if args.xlabel:
		plt.xlabel(args.xlabel)
	if args.ylabel:
		plt.ylabel(args.ylabel)
	if args.title:
		plt.title(args.title)

	if args.logx:
		plt.xscale('log')
	if args.logy:
		plt.yscale('log')

	if args.ylim is not None:
		ylo, yhi = [float(x) for x in args.ylim.split(',')]
		plt.ylim(ylo, yhi)

	# FIXME: code-dupe
	if args.xlim is not None:
		xlo, xhi = [float(x) for x in args.xlim.split(',')]
		plt.xlim(xlo, xhi)

	xgeom, ygeom = [float(s) for s in args.geometry.split(',')]
	plt.gcf().set_size_inches(xgeom, ygeom, forward=True)

	if args.outfile:
		plt.savefig(args.outfile, dpi=args.dpi, bbox_inches=args.bbox_inches)
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
	barparser.add_argument('-s', "--stack", action="store_const",
	                       default=False, const=True, help="stack bars instead"
	                       " of grouping them")

	for p in [lineparser, barparser]:
		p.add_argument('-L', "--legend", type=str, help="labels for legend")

	cdfparser = subparsers.add_parser("cdf", help="draw cumulative distribution")
	cdfparser.set_defaults(plotmode=plot_cdf)

	timechartparser = subparsers.add_parser("tc", help="draw timechart")
	timechartparser.set_defaults(plotmode=plot_timechart)

	for p in [histparser, cdfparser]:
		p.add_argument('-b', "--bins", dest="nbins", type=int, metavar="NBINS",
		               default=15, help="number of bins (default 15)")
		p.add_argument('-a', "--absolute", dest="norm", action="store_const",
		               const=False, default=True, help="Don't normalize y-axis")
		p.add_argument('-l', "--log", action="store_const",
		               default=False, const=True, help="logarithmic histogram")
		p.add_argument('-r', "--range", type=str, help="range of histogram bins"
		               " (min,max)")
		p.add_argument('-p', "--percentile", type=float,  metavar="PCT",
		               default=100.0, help="ignore datapoints beyond PCT percentile")

	timechartparser.add_argument('-d', "--duration", action="store_const",
	                             const=True, default=False, help="read data "
	                             "items as (label, start, length) instead of "
	                             "default (label, start, end)")

	mainargs = [(('-t', "--title"), dict(type=str, help="plot title")),
	            (('-x', "--xlabel"), dict(type=str, help="x-axis label")),
	            (('-X', "--xlim"), dict(type=str, help="x-axis bounds")),
	            (('-y', "--ylabel"), dict(type=str, help="y-axis label")),
	            (('-Y', "--ylim"), dict(type=str, help="y-axis bounds")),
	            (('-c', "--colormap"), dict(type=str, help="pyplot color map")),
	            (('-T', "--tight"),
	             dict(dest="bbox_inches", action="store_const", default=None,
	                  const="tight", help="tight bounding box on output files")),
	            (('-o', "--outfile"),
	             dict(type=str, help="file to save plot in (default none)")),
	            (('-r', "--dpi"),
	             dict(type=int, help="resolution of output file (dots per inch)")),
	            (('-g', "--geometry"),
	             dict(type=str, default="8,6",
	                  help="figure geometry in X,Y format (inches)")),
	            (('-A', "--logx"),
	             dict(action="store_const", const=True, default=False,
	                  help="use logarithmic X axis")),
	            (('-B', "--logy"),
	             dict(action="store_const", const=True, default=False,
	                  help="use logarithmic Y axis")),
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
