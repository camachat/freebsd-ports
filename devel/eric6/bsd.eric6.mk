# Created by: Boris Samorodov <bsam@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$
=======
>>>>>>> upstream/main
#
# This is a set of Makefile macros that are common between devel/eric6 and
# localized language packages for devel/eric6:
# german/eric6, russian/eric6, devel/es-eric6

PORTNAME=	eric6
<<<<<<< HEAD
PORTVERSION=	21.3
=======
PORTVERSION=	21.4
>>>>>>> upstream/main
PORTREVISION?=	0
MASTER_SITES=	SF/eric-ide/${PORTNAME}/stable/${PORTVERSION}
WRKSRC=		${WRKDIR}/${PORTNAME}-${PORTVERSION}

MAINTAINER=	fluffy@FreeBSD.org
COMMENT=	Full featured Python and Ruby editor and IDE based on Qt

LICENSE=	GPLv3

DESCR=		${.CURDIR}/../../devel/eric6/pkg-descr
MAKE_ENV+=	"HOME=${WRKDIR}"

NO_ARCH=	yes
NO_BUILD=	yes
