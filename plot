#!/usr/bin/python

import matplotlib.pyplot as plt

import sys
import argparse

# Expects list of (x, y) tuples or single y-values
def plot_line(lines):
	if len(lines[0]) == 1:
		ys = [float(l[0]) for l in lines]
		xs = range(0, len(ys))
	else:
		xs = [float(l[0]) for l in lines]
		ys = [float(l[1]) for l in lines]
	plt.plot(xs, ys)

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
	values = [float(l[1]) for l in lines]
	w = 0.8
	try:
		plt.bar(range(0, len(lines)), values, width=w, color=args.color)
	except ValueError:
		sys.stderr.write("Bar plot error, perhaps '%s' is an invalid "
		                 "color?\n" % args.color)
		sys.exit(1)

	plt.xticks([i + w/2 for i in range(0, len(lines))], labels,
	           rotation=args.labelangle, rotation_mode="anchor",
	           ha=["center", "right", "left"][cmp(args.labelangle, 0.0)])
	plt.xlim(w-1, len(lines))

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
	if args.live:
		lines = [sys.stdin.readline()]
	else:
		lines = sys.stdin.readlines()

	if ',' in lines[0]:
		splitfn = lambda l: l.split(',')
	else:
		splitfn = lambda l: l.split()

	lines = [splitfn(l) for l in lines]

	if args.xlabel:
		plt.xlabel(args.xlabel)
	if args.ylabel:
		plt.ylabel(args.ylabel)
	if args.title:
		plt.title(args.title)

	args.plotmode(lines)

	if args.ylim is not None:
		ylo, yhi = [int(x) for x in args.ylim.split(',')]
		plt.ylim(ylo, yhi)

	if args.live:
		plt.ion()
		plt.draw()
		while True:
			newline = sys.stdin.readline()
			sys.stderr.write("got new line: %s" % newline)
			if newline == '':
				return
			lines.append(splitfn(newline))
			args.plotmode(lines)
			plt.draw()

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

	scatterparser = subparsers.add_parser("scatter", help="draw scatter plot")
	scatterparser.set_defaults(plotmode=plot_scatter)

	histparser = subparsers.add_parser("hist", help="draw histogram")
	histparser.set_defaults(plotmode=plot_hist)

	barparser = subparsers.add_parser("bar", help="draw bar chart")
	barparser.set_defaults(plotmode=plot_bars)
	barparser.add_argument('-c', "--color", type=str, help="color of plotted bars")
	barparser.add_argument('-r', "--angle", dest="labelangle", type=float,
	                       help="label angle (degrees)", default=0.0)

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
	            (('-l', "--live"),
	             dict(action="store_const", const=True, default=False,
	                  help="update plot as data appears"))]

	for args, kwargs in mainargs:
		mainparser.add_argument(*args, **kwargs)

	args = mainparser.parse_args()

	main()
