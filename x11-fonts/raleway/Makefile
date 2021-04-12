# Created by: Florian Limberger <flo@snakeoilproductions.net>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	raleway
DISTVERSION=	g20140205
PORTREVISION=	1
CATEGORIES=	x11-fonts
=======

PORTNAME=	raleway
DISTVERSION=	4.101
CATEGORIES=	x11-fonts
# The .otf files are no longer comitted to the repository,
# so we need to fetch a release tarball. This is not supported
# by the current implementation of USE_GITHUB, so we overwrite
# the master site here.
MASTER_SITES=	https://github.com/${GH_ACCOUNT}/${GH_PROJECT}/releases/download/${DISTVERSION}/
DISTNAME=	Raleway-${DISTVERSION}
>>>>>>> upstream/main

MAINTAINER=	flo@snakeoilproductions.net
COMMENT=	Elegant sans-serif font, designed in a single thin weight

LICENSE=	OFL11
<<<<<<< HEAD
LICENSE_FILE=	${WRKSRC}/Open\ Font\ License.markdown

USES=		fonts
USE_GITHUB=	yes
GH_ACCOUNT=	theleagueof
GH_TAGNAME=	15dd22b
=======
LICENSE_FILE=	${WRKSRC}/OFL.md

USES=		fonts tar:xz
USE_GITHUB=	nodefault
GH_ACCOUNT=	theleagueof
>>>>>>> upstream/main

NO_ARCH=	yes
NO_BUILD=	yes

<<<<<<< HEAD
PLIST_FILES=	${FONTSDIR}/"Raleway Thin.otf"

do-install:
	${MKDIR} ${STAGEDIR}${FONTSDIR}
	${INSTALL_DATA} ${WRKSRC}/"Raleway Thin.otf" ${STAGEDIR}${FONTSDIR}
=======
do-install:
	${MKDIR} ${STAGEDIR}${FONTSDIR}
	${INSTALL_DATA} ${WRKSRC}/static/OTF/*.otf ${STAGEDIR}${FONTSDIR}
>>>>>>> upstream/main

.include <bsd.port.mk>
