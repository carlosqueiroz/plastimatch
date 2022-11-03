Debian packaging
================
This section describes the recommended build configuration for 
building and packaging the official source tarballs 
and debian packages.

References:
  
#. http://oar.imag.fr/wiki:debian_packaging
#. https://wiki.debian.org/GridenginePackaging/GitGuide
#. http://linux.lsdev.sil.org/wiki/index.php/Packaging_using_gbp
#. https://wiki.debian.org/Diaspora/Packaging/origsource
#. https://wiki.debian.org/PackagingWithGit
#. http://honk.sigxcpu.org/projects/git-buildpackage/manual-html/gbp.import.html


Setting up a build system for the first time
--------------------------------------------
#. Set DEBEMAIL and DEBFULLNAME environment variables (see http://www.debian.org/doc/manuals/maint-guide/first.en.html)

#. Install the requisite packages::

     sudo apt-get install devscripts pbuilder debhelper \
     git-buildpackage cowbuilder

   Note: if your host is not sid, you might need to install a newer gcc version 
   than exists on the native release.  You can do something like this:

      http://lektiondestages.blogspot.com/2013/05/installing-and-switching-gccg-versions.html

#. Make and register ssh keys::

     ssh-keygen -t rsa -C "gregsharp.geo@yahoo.com" -b 4096 -f ~/.ssh/id_rsa_salsa

   Be sure to set up your ~/.ssh/config file to tell it where to find the key::

     # Add this to ~/.ssh/config
     Host salsa.debian.org
             IdentityFile ~/.ssh/id_rsa_salsa

   Then go to https://salsa.debian.org/profile/keys to register the public key.  Wait up to one hour for the key to be registered.

#. Download the relevant directory from the debian-med repository::

     gbp clone --all --pristine-tar git@salsa.debian.org:med-team/plastimatch.git
     
#. Initial setup of pbuilder environment::

     sudo apt-get install debian-archive-keyring
     git-pbuilder create

   Install packages into git-pbuilder.  This saves time when running
   multiple times::

     git-pbuilder login --save-after-login
     apt-get update
     apt-get install \
       apt-utils fakeroot debhelper \
       cmake \
       libblas-dev liblapack-dev libsqlite3-dev \
       libdcmtk-dev libdlib-dev libfftw3-dev \
       libinsighttoolkit5-dev \
       libpng-dev libtiff-dev uuid-dev zlib1g-dev

   See this link for more information https://wiki.debian.org/git-pbuilder


Step 1: Test the debian build
-----------------------------
#. Download the candidate tarball and repack into correct version name::

     export V=1.7.4
     cd ~/debian-med
     wget https://gitlab.com/plastimatch/plastimatch/-/archive/master/plastimatch-master.tar.gz
     tar xvf plastimatch-master.tar.gz
     rm plastimatch-master.tar.gz
     mv plastimatch-master plastimatch-${V}
     tar cvfz plastimatch-${V}.tar.gz plastimatch-${V}
     rm -rf plastimatch-${V}

   Note that the old version number should be used, as we have not yet updated
   the version in the debian changelog.

#. Run debian repacking::

     (cd plastimatch && mk-origtargz ../plastimatch-${V}.tar.gz)

#. Unzip the created tarball, and copy over the debian directory::

     tar xvf plastimatch_${V}+dfsg.1.orig.tar.[xg]z
     cp -r plastimatch/debian plastimatch-${V}

#. Refresh your pbuilder environment (if needed)::

     sudo pbuilder update

#. Run debuild and build::

     cd plastimatch-${V}
     debuild -i -us -uc -S
     sudo pbuilder build ../plastimatch_*.dsc

#. Run lintian on package::

     cd ~/debian-med
     lintian -i *.changes
   
Step 2: Build the tarball
-------------------------
Follow instructions in :ref:`making_a_tarball`.

Step 3: Build the debian package
--------------------------------
#. Clean pbuilder environment (if needed)::

     pbuilder clean

#. Refresh your git-pbuilder environment (if needed)::

     git-pbuilder update

#. Don't edit changelog yet

#. Patch git with upstream::

     gbp import-orig --pristine-tar --uscan 

#. The above won't work if you already edited and committed the
   debian changelog.  Instead, download and then patch.::

     uscan --verbose --force-download
     gbp import-orig --pristine-tar ../plastimatch_1.6.5+dfsg.1.orig.tar.gz
     
#. Update changelog (in an terminal, not emacs)::

     V=1.9.4              # New version
     cd plastimatch
     dch -v ${V}+dfsg.1-1
     git commit -a -m "Update changelog"

   Don't forget to change release status to "unstable"
     
#. Test::

     gbp buildpackage --git-pbuilder --git-ignore-new -j4 --git-postbuild='lintian -i $GBP_CHANGES_FILE'
   
#. If you need select a patch from git, do somthing like this::

     git format-patch HEAD~

#. Push changes to server::

     git push --all origin && git push --tags origin

Various hints
-------------

Switching between git branches
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Like this::

  git checkout pristine-tar
  git checkout upstream
  git checkout master

Full reset of repository
^^^^^^^^^^^^^^^^^^^^^^^^
Like this::

     git checkout pristine-tar
     git reset --hard origin/pristine-tar --
     git checkout upstream
     git reset --hard origin/upstream --
     git checkout master
     git reset --hard origin/master --
     git tag -d upstream/1.6.5+dfsg


Alternatives to running gbp buildpackage
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#. Run gbp buildpackage to create the dsc::

     gbp buildpackage --git-ignore-new -uc -us -j16

   If the host does not contain all needed packages you will need to use pbuilder::

     gbp buildpackage --git-pbuilder --git-ignore-new -uc -us -j16
     
   All the junk that gbp buildpackage makes, such as the orig.tar.gz and the 
   dsc file, gets put in the parent directory.

#. If you want to clean the git directory, you can run::

     debuild clean

#. Test with pbuilder::

     gbp buildpackage --git-pbuilder --git-ignore-new -j16

Rebuilding an existing debian source package
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Like this::

  apt-get source foo
  cd foo-0.0.1
  sudo apt-get build-dep foo
  debuild -i -us -uc -b

See: https://wiki.debian.org/HowToPackageForDebian
