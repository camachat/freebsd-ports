<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	molsketch
DISTVERSION=	0.7.1
DISTVERSIONSUFFIX=	-src
CATEGORIES=	science
MASTER_SITES=	SF/${PORTNAME}/Molsketch/Oxygen-16%20${DISTVERSION}/
=======
PORTNAME=	molsketch
DISTVERSION=	0.7.2
DISTVERSIONSUFFIX=	-src
CATEGORIES=	science
MASTER_SITES=	SF/${PORTNAME}/Molsketch/
>>>>>>> upstream/main
DISTNAME=	Molsketch-${DISTVERSIONFULL}

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	2D molecule editor

LICENSE=	GPLv2
LICENSE_FILE=	${WRKSRC}/COPYING

LIB_DEPENDS=	libopenbabel.so:science/openbabel

<<<<<<< HEAD
USES=		cmake:noninja compiler:c++11-lang desktop-file-utils localbase:ldflags \
=======
USES=		cmake:noninja compiler:c++14-lang desktop-file-utils localbase:ldflags \
>>>>>>> upstream/main
		pkgconfig qt:5 shared-mime-info # build fails with ninja, see https://sourceforge.net/p/molsketch/bugs/41/
USE_QT=		core gui network printsupport svg widgets buildtools_build linguisttools_build qmake_build
USE_LDCONFIG=	${PREFIX}/lib/molsketch

CMAKE_ARGS=	-DMSK_PREFIX:STRING=${PREFIX}

WRKSRC=		${WRKDIR}/Molsketch-${DISTVERSION}

OPTIONS_DEFINE=	DOCS

PORTDOCS=	*

.include <bsd.port.mk>
