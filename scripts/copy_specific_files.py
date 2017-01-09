
import os
import shutil
import argparse


def bool_copy_file(file_name, list_pattern_to_avoid):
    """ Indicates if the file needs to be copied or not

    Details : Check if any pattern of list_pattern_to_avoid is in file_name

    args : - file_name (str) : name of the file under scrutiny
           - list_pattern_to_avoid (list of str) : patterns to avoid

    return: False if a pattern to be avoided is in the file_name
            True otherwise
    """
    for pattern in list_pattern_to_avoid:
        if pattern in file_name:
            return False
    return True


def copy_files(source_dir, target_dir, list_pattern_to_avoid):
    """ Copy all files from source_dir to target_dir
    avoiding files matching any patterns in list_pattern_to_avoid

    args : - source_dir (str) : initial directory
           - target_dir (str) : new directory to be created
           - list_pattern_to_avoid (list) : patterns to avoid

    Details : use topdown = True for topdown approach.
    Walk the subdirectories in source_dir and only copy the relevant files

    """

    # Create target directory if needed
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    # First copy the skeleton of the file (i.e make an empty copy of all the
    # subfolders)
    for root, subFolders, files in os.walk(source_dir, topdown=True):
        folder_path = root[len(source_dir):]
        if (len(folder_path) != 0 and not os.path.exists(target_dir + folder_path)):
            os.makedirs(target_dir + folder_path)

    for root, subFolders, files in os.walk(source_dir, topdown=True):
        folder_path = root[len(source_dir):]
        if (len(folder_path) != 0 and folder_path[-1] != '/'):
            folder_path += '/'
        for file_name in files:
            if bool_copy_file(file_name, list_pattern_to_avoid):
                shutil.copyfile(
                    source_dir +
                    folder_path +
                    file_name,
                    target_dir +
                    folder_path +
                    file_name)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Copy all files from a source dir to a target dir\
         avoiding files with a file name containing specific patterns")
    parser.add_argument("source_dir", help="Source directory", type=str)
    parser.add_argument("target_dir", help="Target directory", type=str)
    parser.add_argument(
        "list_pattern_to_avoid",
        help="List of patterns to avoid",
        nargs='+',
        type=str)
    args = parser.parse_args()

    copy_files(args.source_dir, args.target_dir, args.list_pattern_to_avoid)
