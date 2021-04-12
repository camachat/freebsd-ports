<<<<<<< HEAD
# $FreeBSD$

PORTREVISION=	7
=======
PORTREVISION=	8
>>>>>>> upstream/main
CATEGORIES=	graphics

COMMENT=	GStreamer OpenExr codec plugin

USES+=		compiler:c++11-lang
GST_PLUGIN=	openexr
DIST=		bad

<<<<<<< HEAD
=======
CPPFLAGS+=	-DInt64=uint64_t

>>>>>>> upstream/main
pre-configure:
		${REINPLACE_CMD} -e 's/c++98/c++11/' ${WRKSRC}/ext/openexr/Makefile.in

MASTERDIR=	${.CURDIR}/../../multimedia/gstreamer1-plugins

.include "${MASTERDIR}/Makefile"
