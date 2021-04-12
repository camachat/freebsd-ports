# Created by: Jan Stocker <Jan.Stocker@t-online.de>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	egoboo
PORTVERSION=	2.7.5
PORTREVISION=	11
PORTEPOCH=	1
CATEGORIES=	games
MASTER_SITES=	SF/${PORTNAME}/Linux%20Source%20Distro/${PORTVERSION}
DISTFILES=	${DISTNAME}${EXTRACT_SUFX} \
		${DISTNAME:S/-/-data-/}${EXTRACT_SUFX}
=======

PORTNAME=	egoboo
PORTVERSION=	2.8.1
PORTEPOCH=	1
CATEGORIES=	games
MASTER_SITES=	SF/${PORTNAME}/Linux%20Source%20Distro/${PORTVERSION}%20(Stable)
>>>>>>> upstream/main

MAINTAINER=	amdmi3@FreeBSD.org
COMMENT=	3D dungeon crawling adventure in the spirit of NetHack

<<<<<<< HEAD
USES=		dos2unix gl gmake sdl
USE_SDL=	sdl ttf mixer image
USE_GL=		glu
DOS2UNIX_GLOB=	*.c

WRKSRC=		${WRKDIR}/${DISTNAME}/game
MAKEFILE=	Makefile.unix
MAKE_ARGS=	PREFIX="${STAGEDIR}${PREFIX}"

PLIST_FILES=	bin/${PORTNAME} libexec/${PORTNAME}
PORTDATA=	*

post-patch:
	@${REINPLACE_CMD} -e 's|/usr/share/egoboo|${DATADIR}|g; \
		s|cp -a|cp -RPp|; s|/usr/libexec|${PREFIX}/libexec|' \
		${WRKSRC}/egoboo.sh

post-install:
	@cd ${WRKDIR}/${DISTNAME:S/-/-data-/} && \
		${COPYTREE_SHARE} . ${STAGEDIR}${DATADIR}/
	${STRIP_CMD} ${STAGEDIR}${PREFIX}/libexec/${PORTNAME}
=======
LICENSE=	GPLv3+
LICENSE_FILE=	${WRKSRC}/license.txt

LIB_DEPENDS=	libphysfs.so:devel/physfs \
		libenet.so:net/enet

USES=		dos2unix gl gmake sdl
USE_SDL=	sdl ttf mixer image
USE_GL=		gl glu
DOS2UNIX_GLOB=	Makefile *.c

PROJ_NAME=	egoboo

BUILD_WRKSRC=	${WRKSRC}/src/game
MAKE_ENV=	PROJ_NAME=${PROJ_NAME}
ALL_TARGET=	${PROJ_NAME}

ETCDIR=		${PREFIX}/etc/${PROJ_NAME}
DATADIR=	${PREFIX}/share/${PROJ_NAME}

PORTDOCS=	*
PORTDATA=	*

DESKTOP_ENTRIES="Egoboo" \
		"" \
		"${DATADIR}/icon.bmp" \
		"${PROJ_NAME}" \
		"Game;AdventureGame;RolePlaying;" \
		""

OPTIONS_DEFINE=	DOCS

do-install:
	${INSTALL_PROGRAM} ${BUILD_WRKSRC}/${PROJ_NAME} ${STAGEDIR}${PREFIX}/bin/
	@cd ${WRKSRC} && ${COPYTREE_SHARE} "basicdat modules" ${STAGEDIR}${DATADIR}/
	@${MKDIR} ${STAGEDIR}${ETCDIR}
	${INSTALL_DATA} ${WRKSRC}/controls.txt ${STAGEDIR}${ETCDIR}/controls.txt.sample
	${INSTALL_DATA} ${WRKSRC}/setup.txt ${STAGEDIR}${ETCDIR}/setup.txt.sample

do-install-DOCS-on:
	@cd ${WRKSRC} && ${COPYTREE_SHARE} "BUGS.txt Changelog.txt doc" \
		${STAGEDIR}${DOCSDIR} "-not -empty"
>>>>>>> upstream/main

.include <bsd.port.mk>
