#!/usr/bin/perl
<<<<<<< HEAD
#
# $FreeBSD$
=======
>>>>>>> upstream/main

use strict;
use warnings;
use File::Temp qw(tempfile);

my ($fh, $fn) = tempfile("check-XXXXXX", SUFFIX => '.tmp', TMPDIR => 1, UNLINK => 1);
#print "$fn\n";
