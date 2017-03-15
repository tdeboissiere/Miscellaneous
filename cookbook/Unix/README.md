# Unix cookbook

## Number of lines in file

    wc -l filename

## Number of files in directory

    ls | wc -l

## Find files recursively with certain extension

    find . -type f -name '*.txt'

## Wildcard on expr with parenthesis

    ls *\(blabla\)*

## Configuring ssh

### Definitions

remote-host is something like <username>@IPaddres
For instance user@000.00.000.000

### Step 0: Install openssh on remote machine

sudo apt-get instal openssh-server


### Step 1: Create public and private keys using ssh-key-gen on local-host

jsmith@local-host$ [Note: You are on local-host here]

jsmith@local-host$ ssh-keygen Generating public/private rsa key pair.
Enter file in which to save the key (/home/jsmith/.ssh/id_rsa):[Enter key]
Enter passphrase (empty for no passphrase): [Press enter key]
Enter same passphrase again: [Pess enter key]
Your identification has been saved in /home/jsmith/.ssh/id_rsa.
Your public key has been saved in /home/jsmith/.ssh/id_rsa.pub.
The key fingerprint is:
33:b3:fe:af:95:95:18:11:31:d5:de:96:2f:f2:35:f9 jsmith@local-host


### Step 2: Copy the public key to remote-host using ssh-copy-id


jsmith@local-host$ ssh-copy-id -i ~/.ssh/id_rsa.pub remote-host
jsmith@remote-host's password:
Now try logging into the machine, with "ssh 'remote-host'", and check in:

.ssh/authorized_keys

to make sure we haven't added extra keys that you weren't expecting.

Note: ssh-copy-id appends the keys to the remote-host’s .ssh/authorized_key.
### Step 3: Login to remote-host without entering the password

jsmith@local-host$ ssh remote-host
Last login: Sun Nov 16 17:22:33 2008 from 192.168.1.2
[Note: SSH did not ask for password.]

jsmith@remote-host$ [Note: You are on remote-host here]

### Step 4: Mount relevant remote folders with sshfs in your local machine

Mount with:

sshfs -o delay_connect,reconnect,ServerAliveInterval=5,ServerAliveCountMax=3,allow_other,default_permissions,IdentityFile=/local/path/to/private/key user@000.00.000.000:/home/user /path/your/mount/folder/


Unmount with:

umount user@000.00.000.000:/ &> /dev/null
