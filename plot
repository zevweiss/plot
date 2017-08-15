#!/usr/bin/python

import numpy as np
import matplotlib
from os import getenv
if getenv('DISPLAY'):
	matplotlib.use('TkAgg')
else:
	matplotlib.use('Agg')
import matplotlib.pyplot as plt

import sys
import time
import argparse

cmap = plt.cm.get_cmap()

anim = None
start_time = None
splitfn = None

plot_ymax = None

animation = None

# By default, expects list of tuples of Y values.  With -x, the first
# element of each tuple is instead interpreted as the X value for the
# remaining Y values.
def plot_line(lines):
	global anim

	if args.xypairs and args.xcoord:
		sys.stderr.write("-x and -X conflict for 'line'\n")
		exit(1)

	if args.xypairs:
		if len(lines[0]) < 2 or len(lines[0]) % 2 != 0:
			sys.stderr.write("input malformed for 'line -X'\n")
			exit(1)
		odds = [i for i in range(0, len(lines[0])) if i % 2 == 1]
		evens = [i for i in range(0, len(lines[0])) if i % 2 == 0]
		allxs = [[float(l[i]) for l in lines] for i in evens]
		allys = [[float(l[i]) for l in lines] for i in odds]
	elif args.xcoord:
		if len(lines[0]) < 2:
			sys.stderr.write("'line -x' requires multiple columns\n")
			exit(1)
		allxs = [[float(l[0]) for l in lines]] * (len(lines[0]) - 1)
		allys = [[float(l[c]) for l in lines] for c in range(1, len(lines[0]))]
	else:
		allxs = [list(range(0, len(lines)))] * len(lines[0])
		allys = [[float(l[c]) for l in lines] for c in range(0, len(lines[0]))]

	def mkcolor(i):
		return cmap(float(i) / float(max(len(allys), 2) - 1))
	colors = map(mkcolor, range(len(allys)))

	if args.stack:
		plt.stackplot(allxs[0], allys, colors=colors, labels=args.legend)
	else:
		plotlines = []
		for i, (xs, ys) in enumerate(zip(allxs, allys)):
			kw = {}
			for a in ["markers", "linestyles"]:
				arg = getattr(args, a)
				if arg is not None:
					v = arg[i % len(arg)]
					# HACK.
					kw[a[:-1]] = v

			label = args.legend[i] if args.legend is not None else None
			color = colors[i]
			plotlines += plt.plot(xs, ys, label=label, color=color, **kw)

	if args.legend is not None:
		plt.legend(loc=0)

	if args.live:
		def update_plot(i, lines):
			global start_time
			s = get_inputline()
			newdata = splitfn(s)

			if args.xcoord:
				newx = float(newdata[0])
				newdata = newdata[1:]
			else:
				newx = time.time() - start_time

			if len(newdata) != len(lines):
				sys.stderr.write("got %d data points, expected %d; "
				                 "discarding line: '%s'\n"
				                 % (len(newdata), len(lines), s))
				return lines

			xmax, ymax = 0, 0

			for pl, d in zip(lines, newdata):
				xd, yd = pl.get_data()

				xd = np.append(xd, newx)
				yd = np.append(yd, float(d))

				if args.history > 0:
					if len(xd) > args.history:
						xd = xd[-args.history:]
					if len(yd) > args.history:
						yd = yd[-args.history:]

				xmax = max(xmax, max(xd))
				ymax = max(ymax, max(yd))

				pl.set_data([xd, yd])

			plt.xlim(xd[0], xmax)
			plt.ylim(0, (1.1 * ymax) if ymax != 0.0 else 1)
			plt.draw()

			return lines
		# interval=1 here is to sidestep a bug in MPL (should
		# be zero); looks like a teardown race condition
		anim = animation.FuncAnimation(fig=plt.gcf(), func=update_plot, frames=50,
		                               interval=1, fargs=(plotlines,), blit=True)

def percentile(vals, pct):
	num = min(int((pct/100.0) * len(vals)), len(vals))
	return sorted(vals)[:num]

# expects list of single values
def plot_hist(lines):
	vals = percentile([float(x[0]) for x in lines], args.percentile)
	plt.hist(vals, args.nbins, range=args.range, normed=args.norm, color=cmap(0.5))

# expects a list of sets of values
# FIXME: support different-length columns
def plot_cdf(lines):
	maxes = []

	for i in range(len(lines[0])):
		vals = percentile([float(x[i]) for x in lines], args.percentile)
		maxes.append(max(vals))

		try:
			label = args.legend[i]
		except:
			label = None

		plt.hist(vals, args.nbins, range=args.range, cumulative=True, histtype='step',
		         normed=args.norm, label=label)

	if args.log:
		plt.xscale('log')
	plt.xlim(0, max(maxes))
	plt.ylim(0, 1 if args.norm else len(vals))
	if args.legend is not None:
		plt.legend()

# expects list of (x, y) tuples
def plot_scatter(lines):
	global anim
	xs = [float(l[0]) for l in lines]
	ys = [float(l[1]) for l in lines]
	plt.scatter(xs, ys, marker='x')

	if args.live:
		def update_plot(i):
			s = get_inputline()
			coords = splitfn(s)

			if len(coords) != 2:
				sys.stderr.write("got %d fields, expected 2; "
				                 "discarding line: '%s'\n"
				                 % (len(coords), s))

			x, y = [float(f) for f in coords]
			plt.scatter([x], [y], marker='x')

		anim = animation.FuncAnimation(fig=plt.gcf(), func=update_plot, frames=50,
		                               interval=1)

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
		bars = plt.bar(lefts, thisset, bottom=prevset, width=width, label=label,
		               color=color)

		if args.numbers:
			for b, v in zip(bars, thisset):
				xpos = b.get_x() + (b.get_width() / 2)
				ypos = b.get_height()
				if plot_ymax is not None:
					ypos = min(ypos, plot_ymax)
				plt.text(xpos, ypos, str(int(round(v))), ha="center", va="bottom")

		if args.stack:
			prevset = [x+y for x,y in zip(thisset, prevset)]

	plt.xticks([i + w/2 for i in range(0, len(lines))], labels,
	           rotation=args.labelangle, rotation_mode="anchor",
	           ha=["center", "right", "left"][cmp(args.labelangle, 0.0)])
	plt.xlim(w-1, len(lines))

	if args.legend is not None:
		plt.legend(loc=0)

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
		plt.broken_barh(xdims, ydim, alpha=args.alpha)

	plt.yticks([x + 0.5 for x in xrange(0, len(idnums))], ids)

def plot_heatmap(lines):
	if args.autolabel:
		labels = [l[0] for l in lines]
		values = np.array([[float(f) for f in l[1:]] for l in lines])
	else:
		values = np.array([[float(f) for f in l] for l in lines])

	xlen = len(values[0])

	pc = plt.pcolor(values, cmap=cmap)
	if args.autolabel:
		plt.yticks(np.arange(len(labels))+0.5, labels)

	if args.xlabels is not None:
		plt.xticks(np.arange(xlen)+0.5, args.xlabels,
		           rotation=args.labelangle, rotation_mode="anchor",
		           ha=["center", "right", "left"][cmp(args.labelangle, 0.0)])

	plt.xlim(0, xlen)
	plt.ylim(0, len(values))

	if args.drawlegend:
		cb = plt.colorbar()
		if (args.cblabel):
			cb.set_label(args.cblabel)

# expects small(ish?) set of lines, each of which will be plotted as a
# violin representing the datapoints on that line
def plot_violin(lines):
	if args.autolabel:
		values = [[float(x) for x in l[1:]] for l in lines]
		labels = [l[0] for l in lines]
	else:
		values = [[float(x) for x in l] for l in lines]
	if args.discard != 0:
		trimmed = []
		pct = float(args.discard) / 100.0
		for vs in values:
			drop = int(pct * len(vs))
			vs = sorted(vs)[drop:-drop]
			trimmed.append(vs)
		values = trimmed
	plt.violinplot(values, showmedians=args.medians, showmeans=args.means, showextrema=args.extrema,
	               vert=not args.horizontal)
	if args.autolabel:
		if args.horizontal:
			kwargs = dict(va="center")
		else:
			kwargs = dict(rotation_mode="anchor",
				      ha=["center", "right", "left"][cmp(args.labelangle, 0.0)])
		tickfn = plt.yticks if args.horizontal else plt.xticks
		tickfn(np.arange(len(values))+1, labels, rotation=args.labelangle, **kwargs)

def get_inputline():
	while True:
		s = sys.stdin.readline()
		if s == '' or s[0] != '#':
			return s

def get_input():
	while True:
		s = get_inputline()
		if s == '':
			break
		yield s

def do_plot():
	global cmap, start_time, splitfn, plot_ymax
	if args.live:
		global animation
		import matplotlib.animation as animation
		lines = [get_inputline()]
		start_time = time.time()
	else:
		lines = list(get_input())

	if ',' in lines[0]:
		splitfn = lambda l: l.split(',')
	else:
		splitfn = lambda l: l.split()

	lines = [splitfn(l) for l in lines]

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

	plot_ymax = args.ylim[1]

	args.plotmode(lines)

	if args.xlabel:
		plt.xlabel(args.xlabel)
	if args.ylabel:
		plt.ylabel(args.ylabel)
	if args.title:
		plt.title(args.title)

	if args.logx:
		plt.xscale('log', basex=args.logx)
	if args.logy:
		plt.yscale('log', basey=args.logy)

	plt.xlim(*args.xlim)
	plt.ylim(*args.ylim)

	if args.hline:
		plt.hlines(args.hline, *plt.xlim())
	if args.vline:
		plt.vlines(args.vline, *plt.ylim())

	plt.gcf().set_size_inches(*args.geometry, forward=True)
	plt.gca().set_facecolor(args.background)

	if args.window_title:
		plt.gcf().canvas.set_window_title(args.window_title)

	if args.outfile:
		plt.savefig(args.outfile, dpi=args.dpi, bbox_inches=args.bbox_inches)
	else:
		plt.show()

def main():
	mainparser = argparse.ArgumentParser(description="plot data from stdin")

	subparsers = mainparser.add_subparsers(description="Available plot types:",
					       metavar="PLOTTYPE")

	# Add args to parser -- args is a list of (base, opt) tuples, where base
	# is a (shortopt, longopt, help) tuple and opt is a dict of add_argument()
	# kwargs.
	def add_args(parser, args):
		for base, opt in args:
			s, l, h = base
			flags = (('-'+s,) if s is not None else ()) + ("--"+l,)
			parser.add_argument(*flags, help=h, **opt)

	# helper to create one of the above tuples
	def arg(s, l, h, **kwargs):
		return ((s, l, h), kwargs)

	# further helper for the common boolean case
	def boolarg(s, l, h, **kwargs):
		return arg(s, l, h, action="store_true", **kwargs)

	def add_subcmd(cmd, func, desc, args=[]):
		# python 2.7 doesn't support aliases in add_parser, sadly.
		parser = subparsers.add_parser(cmd, help=desc)
		parser.set_defaults(plotmode=func)
		add_args(parser, args)
		return parser

	def listparser(itemparser, allow_missing=False, reqlen=None):
		def parser(arg):
			if allow_missing:
				p = lambda s: None if s == '' else itemparser(s)
			else:
				p = itemparser
			try:
				v = [p(e) for e in arg.strip().split(',')]
				if reqlen is not None:
					assert len(v) == reqlen
				return v
			except:
				msg = 'invalid argument "%s"' % arg
				count = "%d " % reqlen if reqlen is not None else ""
				count += "[optional] " if allow_missing else ""
				msg += " (should be comma-separated list of %s%ss)" \
				       % (count, itemparser.__name__)
				raise argparse.ArgumentTypeError(msg)
		return parser

	line = add_subcmd("line", plot_line, "draw line plot",
	                  [boolarg('x', "xcoord", "use first column as X coordinates"),
	                   boolarg('X', "xypairs", "use even columns as X coordinates, odd columns as Y"),
	                   arg('m', "markers", "marker styles"),
	                   arg('s', "linestyles", "line styles")])

	scatter = add_subcmd("scatter", plot_scatter, "draw scatter plot")

	hist = add_subcmd("hist", plot_hist, "draw histogram")

	bar = add_subcmd("bar", plot_bars, "draw bar chart",
	                 [boolarg('n', "numbers", "show numeric values on top of each bar")])

	cdf = add_subcmd("cdf", plot_cdf, "draw cumulative distribution")

	tc = add_subcmd("tc", plot_timechart, "draw timechart",
	                [arg('a', "alpha", "opacity of timespan blocks", type=float),
	                 boolarg('d', "duration", "read data items as (label, start, length)"
	                         " instead of default (label, start, end)")])

	heatmap = add_subcmd("heatmap", plot_heatmap, "draw heat map",
	                     [boolarg('l', "autolabel", "use first column as Y-axis labels"),
	                      arg('X', "xlabels", "X-axis labels", type=listparser(str)),
	                      boolarg('L', "drawlegend", "draw legend"),
	                      arg('Z', "cblabel", "colorbar label")])

	violin = add_subcmd("violin", plot_violin, "draw violin plot",
	                    [arg('D', "discard", "discard data points in highest and lowest N percentiles",
	                         type=int, default=0),
	                     boolarg('d', "medians", "show medians"),
	                     boolarg('n', "means", "show means"),
	                     boolarg('a', "extrema", "show extrema"),
	                     boolarg('H', "horizontal", "create horizontal plot"),
			     boolarg('l', "autolabel", "use first column as X-axis labels")])

	for p in [bar, heatmap, violin]:
		add_args(p, [arg('r', "angle", "X-axis label angle (degrees)",
		                 dest="labelangle", type=float, default=0.0)])

	for p in [line, bar, cdf]:
		add_args(p, [arg('L', "legend", "labels for legend")])

	for p in [line, bar]:
		add_args(p, [boolarg('a', "stack", "stack bars instead of grouping them")])

	for p in [hist, cdf]:
		add_args(p, [arg('b', "bins", "number of bins (default 15)",
		                 dest="nbins", type=int, metavar="NBINS", default=15),
		             boolarg('a', "absolute", "Don't normalize y-axis", dest="norm"),
		             boolarg('l', "log", "logarithmic histogram"),
		             arg('r', "range", "range of histogram bins (min,max)",
				 type=listparser(float, reqlen=2)),
		             arg('p', "percentile", "ignore datapoints beyond PCT percentile",
		                 type=float,  metavar="PCT", default=100.0)])

	# "[LO],[HI]" would be preferable here instead of "(LO),(HI)", but that
	# unfortunately causes something in argparse to barf when formatting
	# help output.
	axlim = dict(default=(None,None), metavar="(LO),(HI)",
		     type=listparser(float, allow_missing=True, reqlen=2))
	mainargs = [arg('t', "title", "plot title"),
	            arg('x', "xlabel", "x-axis label"),
	            arg('y', "ylabel", "y-axis label"),
	            arg('X', "xlim", "x-axis bounds", **axlim),
	            arg('Y', "ylim", "y-axis bounds", **axlim),
	            arg('o', "outfile", "file to save plot in (default none)"),
	            arg('c', "colormap", "pyplot color map"),
	            arg('b', "background", "background color", metavar="COLOR", default="white"),
	            arg('T', "tight", "tight bounding box on output files", dest="bbox_inches",
	                action="store_const", const="tight"),
	            arg('r', "dpi", "resolution of output file (dots per inch)", type=int),
	            arg('g', "geometry", "figure geometry in X,Y format (inches)",
	                type=listparser(float, reqlen=2), metavar="X,Y", default=[8.0, 6.0]),
	            arg('A', "logx", "use logarithmic X axis with given base", type=int,
	                metavar="BASE"),
	            arg('B', "logy", "use logarithmic Y axis with given base", type=int,
	                metavar="BASE"),
	            boolarg('l', "live", "update plot as data appears"),
	            arg('H', "history", "number of samples to retain in live mode", type=int,
	                default=0),
	            arg(None, "hline", "Y-positions of horizontal lines to draw across plot",
	                type=listparser(float), metavar="Y1[,Y2...]"),
	            arg(None, "vline", "X-positions of vertical lines to draw across plot",
	                type=listparser(float), metavar="X1[,X2...]"),
	            arg('W', "window-title", "title of plot window", metavar="TITLE")]

	add_args(mainparser, mainargs)

	global args
	args = mainparser.parse_args()

	if not args.outfile and not getenv('DISPLAY'):
		sys.stderr.write("No output file specified but DISPLAY not set\n")
		exit(1)

	do_plot()

if __name__ == "__main__":
	main()
