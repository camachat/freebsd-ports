<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	kimageformats
DISTVERSION=	${KDE_FRAMEWORKS_VERSION}
=======
PORTNAME=	kimageformats
DISTVERSION=	${KDE_FRAMEWORKS_VERSION}
PORTREVISION=	1
>>>>>>> upstream/main
CATEGORIES=	graphics kde kde-frameworks

MAINTAINER=	kde@FreeBSD.org
COMMENT=	KF5 library providing support for additional image formats

<<<<<<< HEAD
LIB_DEPENDS=	libHalf.so:graphics/ilmbase
=======
LIB_DEPENDS=	libImath.so:math/Imath
>>>>>>> upstream/main

USES=		cmake compiler:c++11-lib kde:5 pkgconfig qt:5 tar:xz
USE_KDE=	archive ecm
USE_QT=		core gui printsupport widgets \
		buildtools_build qmake_build

<<<<<<< HEAD
=======
CFLAGS+=	-I${LOCALBASE}/include/Imath

>>>>>>> upstream/main
OPTIONS_DEFINE=	AVIF OPENEXR
OPTIONS_DEFAULT=	AVIF OPENEXR
OPTIONS_SUB=	yes

AVIF_DESC=		AV1 Image File Format via libavif
AVIF_LIB_DEPENDS=	libavif.so:graphics/libavif
AVIF_CMAKE_BOOL_OFF=	CMAKE_DISABLE_FIND_PACKAGE_libavif

<<<<<<< HEAD
OPENEXR_LIB_DEPENDS=	libIlmImf.so:graphics/openexr
OPENEXR_CMAKE_BOOL_OFF=	CMAKE_DISABLE_FIND_PACKAGE_OpenEXR

=======
OPENEXR_LIB_DEPENDS=	libOpenEXR.so:graphics/openexr
OPENEXR_CMAKE_BOOL_OFF=	CMAKE_DISABLE_FIND_PACKAGE_OpenEXR

post-patch:
		${REINPLACE_CMD} s,Imf::Int64,uint64_t, \
			${WRKSRC}/src/imageformats/exr.cpp

>>>>>>> upstream/main
.include <bsd.port.mk>
