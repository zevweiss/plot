plot
====

A simple but useful wrapper around some basic pyplot functionality.

Reads data from stdin and pops up a window showing a plot of it.

## Plot types

`plot` can generate the following types of graphs:

- Line graphs (-l, --line): expects (x,y) pairs

- Bar graphs (-b, --bar): expects (label,value) pairs

- Histograms (-g, --histogram): expects single values

- Scatter plot (-s, --scatter): expects (x,y) pairs

## Examples

To show a 20-binned histogram of git commit frequency:

    $ git log --date=short --pretty=tformat:"%ad" | uniq -c | plot --histogram --bins 20

## TODO

- Multi-dataset support

- Label support (e.g. flag to read first line as list of labels)

## Author

Zev Weiss  
<zevweiss@gmail.com>
