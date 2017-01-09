
import os
import argparse


def bool_remove_file(file_name, list_pattern_to_remove):
    """ Indicates if the file needs to be removed or not

    Details : Check if any pattern of list_pattern_to_remove\
              is in file_name

    args : - file_name (str) : name of the file under scrutiny
           - list_pattern_to_remove (list of str) : \
             patterns to remove

    return: True if a pattern to be removed is in the file_name
            False otherwise
    """
    for pattern in list_pattern_to_remove:
        if pattern in file_name:
            return True
    return False


def remove_files(source_dir, list_pattern_to_remove):
    """ Remove all files in source_dir which contain a pattern\
        from list_pattern_to_remove

    args : - source_dir (str) : initial directory
           - list_pattern_to_remove (list) : patterns to remove

    Details : use topdown = True for topdown approach.
    Walk the subdirectories in source_dir and only copy the relevant files

    """

    for root, subFolders, files in os.walk(source_dir, topdown=True):
        folder_path = root[len(source_dir):]
        if (len(folder_path) != 0 and folder_path[-1] != '/'):
            folder_path += '/'
        for file_name in files:
            if bool_remove_file(file_name, list_pattern_to_remove):
                os.remove(source_dir + folder_path + file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Remove all files from a source dir\
        with a file name containing specific patterns")
    parser.add_argument("source_dir", help="Source directory", type=str)
    parser.add_argument(
        "list_pattern_to_remove",
        help="List of patterns to remove",
        nargs='+',
        type=str)
    args = parser.parse_args()

    remove_files(args.source_dir, args.list_pattern_to_remove)
