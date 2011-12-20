#!/usr/bin/python

import matplotlib.pyplot as plt

import sys
import argparse

# Expects list of (x,y) tuples
def plot_line(lines):
	xs = [float(l[0]) for l in lines]
	ys = [float(l[1]) for l in lines]
	plt.plot(xs,ys)
	pass

# expects list of single values
def plot_hist(lines):
	vals = [float(x[0]) for x in lines]
	plt.hist(vals,args.nbins)

# expects list of (x,y) tuples
def plot_scatter(lines):
	xs = [float(l[0]) for l in lines]
	ys = [float(l[1]) for l in lines]
	plt.scatter(xs,ys,marker='x')

# expects list of (label,value) pairs
def plot_bars(lines):
	labels = [l[0] for l in lines]
	values = [float(l[1]) for l in lines]
	w = 0.8
	plt.bar(range(0,len(lines)),values,width=w)
	plt.xticks([i + w/2 for i in range(0,len(lines))],labels)
	pass

# expects a list of single values
def plot_cdf(lines):
	vals = [float(x[0]) for x in lines]
	plt.hist(vals,args.nbins,cumulative=True,histtype='step')
	plt.xlim(0,max(vals))
	plt.ylim(0,len(vals))

# expects list of (label,start,end) tuples
def plot_timechart(lines):
	ids = sorted(set([l[0] for l in lines]))
	idnums = dict([(j,i) for (i,j) in enumerate(ids)])

	for label in ids:
		parts = [l for l in lines if l[0] == label]
		base = idnums[label]
		ydim = (base,1)
		xdims = []
		for p in parts:
			start = float(p[1])
			end = float(p[2])
			xdims.append((start,end-start))
		plt.broken_barh(xdims,ydim,alpha=0.5)

	plt.yticks([x + 0.5 for x in xrange(0,len(idnums))],ids)

def main():
	lines = sys.stdin.readlines()
	if ',' in lines[0]:
		lines = [l.split(',') for l in lines]
	else:
		lines = [l.split() for l in lines]
	args.plotmode(lines)

	if args.outfile:
		plt.savefig(args.outfile)
	else:
		plt.show()

if __name__ == "__main__":
	mainparser = argparse.ArgumentParser(description="plot data from stdin")

	subparsers = mainparser.add_subparsers()

	# python 2.7 doesn't support aliases in add_parser, sadly.
	lineparser = subparsers.add_parser("line",help="draw line plot")
	lineparser.set_defaults(plotmode=plot_line)

	scatterparser = subparsers.add_parser("scatter",help="draw scatter plot")
	scatterparser.set_defaults(plotmode=plot_scatter)

	histparser = subparsers.add_parser("hist",help="draw histogram")
	histparser.set_defaults(plotmode=plot_hist)

	barparser = subparsers.add_parser("bar",help="draw bar chart")
	barparser.set_defaults(plotmode=plot_bars)

	cdfparser = subparsers.add_parser("cdf",help="draw cumulative distribution")
	cdfparser.set_defaults(plotmode=plot_cdf)

	timechartparser = subparsers.add_parser("tc",help="draw timechart")
	timechartparser.set_defaults(plotmode=plot_timechart)

	for p in [histparser,cdfparser]:
		p.add_argument('-b',"--bins",dest="nbins",type=int,metavar="NBINS",
		               help="number of bins (default 15)")
		p.set_defaults(nbins=15)

	mainparser.add_argument('-o',"--outfile",type=str,
	                        help="file to save plot in (default none)")
	mainparser.set_defaults(outfile=None)

	args = mainparser.parse_args()

	main()
