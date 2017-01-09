
import os


def bool_file_exist(list_file):
    """Verify if the files in list_file exist

    Detail:
        Use os.path.isfile to check

    Args:
        liste_file = (list of str) a list of file

    Returns:
        bool

    Raises:
        void
    """

    for filou in list_file:
        if not os.path.isfile(filou):
            print "File: ", filou, "is missing"
            return False
        return True


def COL(string_name, color):
    """
        Attaches color prefixes/suffixes to given string

        Detail:

        Arguments:
        string_name (str) the string name
        color       (str) the desired color

        Outputs:

        returns a string enhanced with color  pref/suff
        """

    if color == 'blue':
        pref = '\033[94m'

    if color == 'header':
        pref = '\033[95m'

    if color == 'green':
        pref = '\033[92m'

    if color == 'warning':
        pref = '\033[93m'

    if color == 'fail':
        pref = '\033[91m'

    suff = '\033[0m'
    return pref + string_name + suff


def print_utility(string_name):
    """
    prints a string, highlighted with *****

    Detail:

    Arguments:
    string_name (str) the string name

    Outputs:

    Highlights a string print output with ****
    """

    print ''
    print '********* ' + string_name + ' ************'
    print ''


def create_directory(path_name):
    """
        Creates a directory with name path_name

        Detail:

        Arguments:
        path_name (str) the full directory path name

        Outputs:

        Creates a directory at the given location
        """
    if not os.path.exists(path_name):
        os.makedirs(path_name)
        print_utility(COL("Creating directory " + path_name, 'blue'))
        return path_name
    return path_name
