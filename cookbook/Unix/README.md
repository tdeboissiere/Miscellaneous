# Unix cookbook

## Number of lines in file

    wc -l filename

## Number of files in directory

    ls | wc -l

## Find files recursively with certain extension

    find . -type f -name '*.txt'
