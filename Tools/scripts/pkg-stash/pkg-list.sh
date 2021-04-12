#!/bin/sh
<<<<<<< HEAD
# $FreeBSD$
=======
>>>>>>> upstream/main

for i in . `make all-depends-list`; do
	cd $i && [ -f "`make -V PKGFILE`" ] && make -V PKGFILE
done
