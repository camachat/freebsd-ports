PORTNAME=	diffpdf
PORTVERSION=	2.1.3
PORTREVISION=	55
CATEGORIES=	graphics
MASTER_SITES=	http://www.qtrac.eu/

MAINTAINER=	dinoex@FreeBSD.org
COMMENT=	Compare two PDF files-textually or visually
WWW=		http://www.qtrac.eu/diffpdf-foss.html

LICENSE=	GPLv2+

BROKEN=		unfetchable
DEPRECATED=	withdrawn from author
EXPIRATION_DATE=	2025-04-15

LIB_DEPENDS=	libpoppler-qt5.so:graphics/poppler-qt5

USES=		compiler:c++11-lang qmake qt:5 gl
USE_GL=		gl
USE_QT=		buildtools linguist:build core gui printsupport widgets

PLIST_FILES=	bin/diffpdf

pre-configure:
	( cd ${BUILD_WRKSRC} && ${LRELEASE} ${BUILD_WRKSRC}/${PORTNAME}.pro )

do-install:
	${INSTALL_PROGRAM} ${WRKSRC}/diffpdf ${STAGEDIR}${PREFIX}/bin/

.include <bsd.port.mk>
