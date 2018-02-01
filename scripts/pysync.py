import os
import shlex
from subprocess import call
import argparse


def sync_to_cloud(sshkey, cloud_home, source_folder, target_folder=None):
    """Sync data from local machine to remote destination on the
    cloud (AWS or GCP)

    All the data specified in local_folder is recursively sync'd

    Args:
        sshkey: (str) absolute path to the sshkey to allow transfer
        cloud_home: (str) absolute home folder path on remote location
        local_folder: (str) path to local folder to sync
        remote_folder: (str) folder name for the sync'd folder in the remote location
    """

    rsync_part = "rsync -avL --progress -e"
    ssh_part = "'ssh -i %s'" % sshkey
    source_folder_part = os.path.abspath(source_folder) + "/"  # Need the '/' to properly sync the folder
    if target_folder is None:
        target_folder_part = os.path.join(cloud_home, os.path.basename(source_folder))
    else:
        target_folder_part = os.path.join(cloud_home, target_folder)

    command = " ".join([rsync_part, ssh_part, source_folder_part, target_folder_part])
    command = shlex.split(command)

    print("Calling: %s" % " ".join(command))

    call(command)


def sync_from_cloud(sshkey, cloud_home, source_folder, target_folder=None):
    """Sync data from remote destination on the
    cloud (AWS or GCP) to local machine

    All the data specified in remote_folder is recursively sync'd

    Args:
        sshkey: (str) absolute path to the sshkey to allow transfer
        cloud_home: (str) absolute home folder path on remote location
        source_folder: (str) path from cloud_home to the folder to sync from the cloud
        target_folder: (str) local path where to store remote_folder
    """

    rsync_part = "rsync -avL --progress -e"
    ssh_part = "'ssh -i %s'" % sshkey
    source_folder_part = os.path.join(cloud_home, source_folder) + "/"  # Need the '/' to properly sync the folder

    if target_folder is None:
        target_folder_part = os.path.join(os.environ.get("HOME"), os.path.basename(source_folder))
    else:
        target_folder_part = os.path.join(os.environ.get("HOME"), target_folder)

    command = " ".join([rsync_part, ssh_part, source_folder_part, target_folder_part])
    command = shlex.split(command)

    print("Calling: %s" % " ".join(command))

    call(command)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Sync files to AWS/GCP')

    #######################
    # General parameters
    #######################
    parser.add_argument('folders',
                        type=str,
                        nargs="+",
                        help=("Enter space separated folders. "
                              "At most 2 arguments are allowed. "
                              "If only one argument is passed, target folder "
                              "will have the same name as the source. "
                              "If 2 arguments are passed, the first one is the source "
                              "and the second one is the target."))

    parser.add_argument('--gcp',
                        action="store_true",
                        help="Sync to GCP cloud")

    parser.add_argument('--aws',
                        action="store_true",
                        help="Sync to AWS cloud")

    parser.add_argument('--get',
                        action="store_true",
                        help="Get data from cloud")

    parser.add_argument('--push',
                        action="store_true",
                        help="Push data to cloud")

    args = parser.parse_args()

    assert len(args.folders) <= 2, "2 arguments max for the folders field"
    assert not(args.gcp and args.aws), "Only sync from one cloud platform at a time"
    assert not(args.get and args.push), "Only sync from one direction at a time"

    if len(args.folders) == 1:
        source_folder = args.folders[0]
        target_folder = None
    else:
        source_folder = args.folders[0]
        target_folder = args.folders[1]

    if args.gcp:

        sshkey = "<path/to/sshkey>"
        cloud_home = "<gcp_home_path>"

    if args.aws:

        sshkey = "<path/to/.pem>"
        cloud_home = "<aws_home_path>"

    if args.get:
        sync_from_cloud(sshkey, cloud_home, source_folder, target_folder)

    elif args.push:
        sync_to_cloud(sshkey, cloud_home, source_folder, target_folder)
