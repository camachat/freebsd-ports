PORTNAME=	atanks
PORTVERSION=	6.5
PORTREVISION=	3
CATEGORIES=	games
MASTER_SITES=	SF/${PORTNAME}/${PORTNAME}/${DISTNAME}/

MAINTAINER=	jessefrgsmith@yahoo.ca
COMMENT=	2D tank game with large, cartoonish weapons
WWW=		http://atanks.sourceforge.net/

LICENSE=	GPLv2+
LICENSE_FILE=	${WRKSRC}/COPYING

LIB_DEPENDS=	liballeg.so:devel/allegro

USES=		compiler:c++0x gmake

PORTDOCS=	Changelog README README_ru.txt credits.txt

OPTIONS_DEFINE=	DOCS

post-patch:
	@${REINPLACE_CMD} -e '/"optionitembase\.h"/ { x; s,^,#include\
		"button.h",; G; }' ${WRKSRC}/src/optionitem.h

post-install:
	${LN} -sf ../../share/icons/hicolor/48x48/apps/${PORTNAME}.png \
		${STAGEDIR}${PREFIX}/share/pixmaps/

post-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${PORTDOCS:S|^|${WRKSRC}/|} ${STAGEDIR}${DOCSDIR}

.include <bsd.port.mk>
