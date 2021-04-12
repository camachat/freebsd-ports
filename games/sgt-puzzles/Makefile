<<<<<<< HEAD
# Created by: Dmitry Marakasov <amdmi3@FreeBSD.org>
# $FreeBSD$

PORTNAME=	puzzles
PORTVERSION=	20201208
=======
PORTNAME=	puzzles
PORTVERSION=	20210409
>>>>>>> upstream/main
CATEGORIES=	games
MASTER_SITES=	https://www.chiark.greenend.org.uk/~sgtatham/puzzles/ \
		https://mirror.amdmi3.ru/distfiles/
PKGNAMEPREFIX=	sgt-
DISTNAME=	${PORTNAME}-${PORTVERSION}.${COMMITHASH}

MAINTAINER=	amdmi3@FreeBSD.org
COMMENT=	Simon Tatham's Portable Puzzle Collection

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENCE

<<<<<<< HEAD
COMMITHASH=	84cb4c6

USES=		autoreconf gmake gnome pkgconfig
GNU_CONFIGURE=	yes
CONFIGURE_ARGS=	--program-prefix=sgt-
INSTALL_TARGET=	install-strip

PORTDOCS=	*

PUZZLE_LIST=	blackbox bridges cube dominosa fifteen filling flip \
		flood galaxies guess inertia keen lightup loopy magnets map \
		mines net netslide palisade pattern pearl pegs range rect \
		samegame signpost singles sixteen slant solo tents towers \
		tracks twiddle undead unequal unruly untangle

.for f in ${PUZZLE_LIST}
PLIST_FILES+=	bin/sgt-${f}
DESKTOP_FILES_PLIST_FILES+=\
		share/pixmaps/sgt-${f}.png
DESKTOP_FILES_DESKTOP_ENTRIES+=\
		"${f}" \
		"${f} puzzle game" \
		"sgt-${f}" \
		"sgt-${f}" \
		"Game;LogicGame;" \
		""
.endfor

OPTIONS_DEFINE=	DESKTOP_FILES DOCS
OPTIONS_DEFAULT=DESKTOP_FILES GTK2
OPTIONS_SINGLE=	GTK
OPTIONS_SINGLE_GTK=	GTK2 GTK3

DESKTOP_FILES_DESC=	Install menu entries (lots of them)

GTK_DESC=		Gtk version
GTK2_CONFIGURE_ON=	--with-gtk=2
GTK2_USE=		GNOME=gtk20
GTK3_CONFIGURE_ON=	--with-gtk=3
GTK3_USE=		GNOME=gtk30

post-patch:
	@${REINPLACE_CMD} -e 's|-Werror||' ${WRKSRC}/configure.ac \
		${WRKSRC}/configure

post-install-DESKTOP_FILES-on:
.for f in ${PUZZLE_LIST}
	${INSTALL_DATA} ${WRKSRC}/icons/${f}-48d24.png ${STAGEDIR}${PREFIX}/share/pixmaps/sgt-${f}.png
.endfor

post-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${WRKSRC}/puzzles.txt ${WRKSRC}/HACKING ${STAGEDIR}${DOCSDIR}/

check-puzzle-list-completeness: extract
	@${ECHO_CMD} ${PUZZLE_LIST} | ${FMT} -w 1 | ${SORT} > ${WRKDIR}/puzzle_list.port
	@${LS} ${WRKSRC}/icons | ${GREP} 48d24.png | ${CUT} -f 1 -d- | ${SORT} > ${WRKDIR}/puzzle_list.upstream
	@cmp ${WRKDIR}/puzzle_list.port ${WRKDIR}/puzzle_list.upstream
=======
COMMITHASH=	69b5e75

BUILD_DEPENDS=	convert:graphics/ImageMagick6-nox11

USES=		cmake gnome perl5 pkgconfig shebangfix
USE_PERL5=	build
SHEBANG_FILES=	icons/square.pl icons/cicon.pl
CMAKE_ARGS=	-DNAME_PREFIX:STRING=sgt-

OPTIONS_SINGLE=	GTK
OPTIONS_SINGLE_GTK=	GTK2 GTK3
OPTIONS_DEFAULT=GTK3

GTK_DESC=	Gtk version
GTK2_CMAKE_ON=	-DPUZZLES_GTK_VERSION=2
GTK2_USE=	GNOME=gtk20
GTK3_CMAKE_ON=	-DPUZZLES_GTK_VERSION=3
GTK3_USE=	GNOME=gtk30
>>>>>>> upstream/main

.include <bsd.port.mk>
