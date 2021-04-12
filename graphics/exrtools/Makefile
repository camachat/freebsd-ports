# Created by: Emanuel Haupt <ehaupt@critical.ch>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	exrtools
PORTVERSION=	0.4
PORTREVISION=	24
=======

PORTNAME=	exrtools
PORTVERSION=	0.4
PORTREVISION=	25
>>>>>>> upstream/main
CATEGORIES=	graphics
MASTER_SITES=	http://scanline.ca/exrtools/ \
		LOCAL/ehaupt

MAINTAINER=	ehaupt@FreeBSD.org
COMMENT=	Utilities for manipulating with HDR images in OpenEXR format

LICENSE=	GPLv2

<<<<<<< HEAD
LIB_DEPENDS=	libImath.so:graphics/ilmbase \
		libpng.so:graphics/png \
		libIlmImf.so:graphics/openexr
=======
LIB_DEPENDS=	libImath.so:math/Imath \
		libpng.so:graphics/png \
		libOpenEXR.so:graphics/openexr
>>>>>>> upstream/main

USES=		compiler:c++11-lang gmake jpeg localbase:ldflags pkgconfig

GNU_CONFIGURE=	yes

post-patch:
	@${REINPLACE_CMD} -e 's|<values\.h>|<float.h>|' \
		${WRKSRC}/src/exrnormalize.c \
		${WRKSRC}/src/exrpptm.c \
		${WRKSRC}/src/exrstats.c

.include <bsd.port.mk>
