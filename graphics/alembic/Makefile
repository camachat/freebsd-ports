# Created by: Alexey Dokuchaev <danfe@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	alembic
PORTVERSION=	1.7.16
PORTREVISION=	2
=======

PORTNAME=	alembic
DISTVERSION=	1.8.0_beta2
>>>>>>> upstream/main
CATEGORIES=	graphics

MAINTAINER=	danfe@FreeBSD.org
COMMENT=	Open framework for storing and sharing graphic scene data

LICENSE=	BSD3CLAUSE

<<<<<<< HEAD
LIB_DEPENDS=	libImath.so:graphics/ilmbase
=======
LIB_DEPENDS=	libImath.so:math/Imath
LIB_DEPENDS+=	libsz.so:science/szip
>>>>>>> upstream/main

USES=		cmake compiler:c++11-lang
USE_CXXSTD=	c++11
USE_GITHUB=	yes
USE_LDCONFIG=	yes

OPTIONS_DEFINE=	HDF5
OPTIONS_DEFAULT=	HDF5
OPTIONS_SUB=	yes

HDF5_LIB_DEPENDS=	libhdf5.so:science/hdf5
HDF5_CMAKE_ON=	-DUSE_HDF5:BOOL=ON

<<<<<<< HEAD
PLIST_FILES=	lib/libAlembic.so.${PORTVERSION} \
		lib/libAlembic.so.${PORTVERSION:R}
=======
PLIST_FILES=	lib/libAlembic.so.${PORTVERSION:R} \
		lib/libAlembic.so.${PORTVERSION:R:R}
>>>>>>> upstream/main

do-test:
	@cd ${TEST_WRKSRC} && ctest -C ${CMAKE_BUILD_TYPE} ${_MAKE_JOBS}

.include <bsd.port.mk>
