PORTNAME=	wxhexeditor
PORTVERSION=	0.24
PORTREVISION=	7
CATEGORIES=	editors
MASTER_SITES=	SF/${PORTNAME}/wxHexEditor/v${PORTVERSION}%20Beta/
DISTNAME=	wxHexEditor-v${PORTVERSION}-src

MAINTAINER=	danfe@FreeBSD.org
COMMENT=	Graphical hex editor for big files with x86 disassembly support
WWW=		https://www.wxhexeditor.org/

LICENSE=	GPLv2

LIB_DEPENDS=	libudis86.so:devel/udis86 \
		libmhash.so:security/mhash

USES=		compiler:c++11-lang gettext-tools gmake tar:xz
USE_WX=		3.0+
MAKE_ARGS=	WXCONFIG="${WX_CONFIG}"

WRKSRC=		${WRKDIR}/wxHexEditor

# Follow the ${MTREE_FILE} which specifies simple, unsuffixed names
# for these particular locales
post-patch:
	@${MV} ${WRKSRC}/locale/hu_HU ${WRKSRC}/locale/hu
	@${MV} ${WRKSRC}/locale/ja_JP ${WRKSRC}/locale/ja
	@${MV} ${WRKSRC}/locale/nl_NL ${WRKSRC}/locale/nl

.include <bsd.port.mk>
