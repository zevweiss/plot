plot
====

A simple but useful wrapper around some basic pyplot functionality.

Reads data from stdin and pops up a window showing a plot of it.  It's
designed to be easy to use with the sort of data that can be munged
into the appropriate format with sed, awk, cut, etc.  Data elements
can be whitespace- or comma-separated.

## Plot types

`plot TYPE` can generate the following `TYPE`s of graphs:

- Line graphs (`line`): expects (x,y) pairs or single y-values

- Bar graphs (`bar`): expects (label,value) pairs

- Histograms (`hist`): expects single values

- Scatter plot (`scatter`): expects (x,y) pairs

- Cumulative distribution (`cdf`): expects single values

- Timechart (`tc`): expects (label,start,end) tuples

## Examples

To show a 20-binned histogram of git commit frequency:

    $ git log --date=short --pretty=tformat:"%ad" | uniq -c | plot hist --bins 20

Scatter plot of line count vs. file size:

    $ wc -lc * | head -n-1 | plot -x Lines -y Bytes scatter

Cumulative distribution of word length

    $ awk '{ for (i=1; i<=NF; i++) print length($i) }' wrnpc.txt | plot cdf -b20

## TODO

- Multi-dataset support

- Label support (e.g. flag to read first line as list of labels)

- pyplot kwargs in command-line arguments

- Logarithmic axes

## Author

Zev Weiss
<zevweiss@gmail.com>
