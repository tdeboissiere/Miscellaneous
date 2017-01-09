
import os
import argparse


def rename_files_dir(target_dir, old_pattern, new_pattern):
    """ Rename all files in target_dir by changing old_pattern to new_pattern

    args : - target_dir (str) : target directory
           - old_pattern (str) : pattern to replace
           - new_pattern (str) : pattern to substitute to old_pattern
    """

    # First rename subFolders. Key point is to use a bottom up approach: use
    # topdown=False
    for root, subFolders, files in os.walk(target_dir, topdown=False):
        for direc in subFolders:
            if old_pattern in direc:
                new_dir = direc.replace(old_pattern, new_pattern)
                old_filepath = os.path.join(root, direc)
                new_filepath = os.path.join(root, new_dir)
                os.rename(old_filepath, new_filepath)

    # Then rename files
    for root, subFolders, files in os.walk(target_dir):
        for indiv_file in files:
            if old_pattern in indiv_file:
                new_file = indiv_file.replace(old_pattern, new_pattern)
                old_filepath = os.path.join(root, indiv_file)
                new_filepath = os.path.join(root, new_file)
                os.rename(old_filepath, new_filepath)


def rename_pattern_in_files(
    target_dir,
    old_pattern,
    new_pattern,
        list_file_types):
    """ Change old_pattern to new_pattern for files in target_dir

    Detail : Look for occurrences of old_pattern INSIDE
    any file type matching an element of list_file_types

    args : - target_dir (str) : target directory
           - old_pattern (str) : pattern to replace
           - new_pattern (str) : pattern to substitute to old_pattern
           - list_file_types (list) : list of file types for which to change
                                    old pattern to new_pattern
    """
    # Rewrite all files in target_dir by changin old_pattern to new_pattern
    # Everytime it is encountered
    for root, subFolders, files in os.walk(target_dir):
        for indiv_file in files:
            # Check if indiv_file is of a file type we want to modify
            if indiv_file.endswith(tuple(list_file_types)):
                fin = open(os.path.join(root, indiv_file), "r")
                stuff = list(fin.readlines())
                fin.close()
                fout = open(os.path.join(root, indiv_file), "w")
                for line in stuff:
                    fout.write(line.replace(old_pattern, new_pattern))
                fout.close()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Substitute (in place) a new pattern for an old pattern\
                     in all files of a given directory")

    parser.add_argument("target_dir", help="Source directory", type=str)
    parser.add_argument(
        "old_pattern",
        help="String pattern to replace",
        type=str)
    parser.add_argument(
        "new_pattern",
        help="String pattern to replace old_pattern",
        type=str)
    parser.add_argument(
        "list_file_types",
        help="List of file types (.py, .C etc) for which \
              to make the pattern substitution",
        nargs='+',
        type=str)

    args = parser.parse_args()

    rename_files_dir(args.target_dir, args.old_pattern, args.new_pattern)
    rename_pattern_in_files(
        args.target_dir,
        args.old_pattern,
        args.new_pattern,
        args.list_file_types)
