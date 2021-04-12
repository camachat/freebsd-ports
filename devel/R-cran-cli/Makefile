# Created by: TAKATSU Tomonari <tota@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	cli
PORTVERSION=	2.3.1
=======

PORTNAME=	cli
PORTVERSION=	2.4.0
>>>>>>> upstream/main
CATEGORIES=	devel
DISTNAME=	${PORTNAME}_${PORTVERSION}

MAINTAINER=	tota@FreeBSD.org
COMMENT=	Helpers for Developing Command Line Interfaces

LICENSE=	MIT

<<<<<<< HEAD
CRAN_DEPENDS=	R-cran-assertthat>0:math/R-cran-assertthat \
		R-cran-glue>0:devel/R-cran-glue
=======
CRAN_DEPENDS=	R-cran-glue>0:devel/R-cran-glue
>>>>>>> upstream/main
BUILD_DEPENDS=	${CRAN_DEPENDS}
RUN_DEPENDS=	${CRAN_DEPENDS}

USES=		cran:auto-plist

post-install:
	@${ECHO_CMD} "@dir ${R_MOD_DIR}/help/figures" >> ${TMPPLIST}

.include <bsd.port.mk>
