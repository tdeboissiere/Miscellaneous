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


    $ ssh-keygen  # Generating public/private rsa key pair.


### Step 2: Copy the public key to remote-host using ssh-copy-id


    $ ssh-copy-id -i ~/.ssh/id_rsa.pub remote-host

## Copy with progbar

rsync -r --info=progress2 source dest

Inspect

    .ssh/authorized_keys

to make sure we haven't added extra keys that you weren't expecting.

### Step 3: Login to remote-host without entering the password

    $ ssh remote-host

### Step 4: Mount relevant remote folders with sshfs in your local machine

Mount with:

    $ sshfs -o delay_connect,reconnect,ServerAliveInterval=5,ServerAliveCountMax=3,allow_other,default_permissions,IdentityFile=/local/path/to/private/key user@000.00.000.000:/home/user /path/your/mount/folder/


Unmount with:

    $ umount user@000.00.000.000:/ &> /dev/null


#### See the alias

    type `<aliascmd>`

### Cat to append

    cat >> file.txt

### ls with absolute path

    readlink -f *.csv | cat > truc.txt

### ls with absolute path

    readlink -f *.csv | cat > truc.txt
    
### recursive print of when files were last accessed

    stat --printf="%y %n\n" $(ls -tr $(find * -type f))

### find files with a given string

    grep -rnw '/path/to/somewhere/' -e "pattern"
