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
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="plot data from stdin")

	parser.add_argument('-l',"--line",dest="plotmode",action="store_const",
	                    const=plot_line,help="draw line plot (default)")
	parser.add_argument('-s',"--scatter",dest="plotmode",action="store_const",
	                    const=plot_scatter,help="draw scatter plot")
	parser.add_argument('-g',"--histogram",dest="plotmode",action="store_const",
	                    const=plot_hist,help="draw histogram")
	parser.add_argument('-b',"--bar",dest="plotmode",action="store_const",
	                    const=plot_bars,help="draw bar chart")
	parser.add_argument('-c',"--cdf",dest="plotmode",action="store_const",
	                    const=plot_cdf,help="draw cumulative distribution function")
	parser.add_argument('-t',"--timechart",dest="plotmode",action="store_const",
	                    const=plot_timechart,help="draw timechart")

	parser.add_argument('-n',"--bins",dest="nbins",type=int,metavar="NBINS",
	                    help="number of bins for histograms/cdfs (default 10)")

	args = parser.parse_args()
	if args.plotmode is None:
		args.plotmode = plot_line
	if args.nbins is None:
		args.nbins = 10

	main()
