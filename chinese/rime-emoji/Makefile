PORTNAME=	rime-emoji
DISTVERSION=	g20241102
CATEGORIES=	chinese textproc

MAINTAINER=	khng@FreeBSD.org
COMMENT=	Rime emoji suggestions support
WWW=		https://github.com/rime/rime-emoji

LICENSE=	LGPL3
LICENSE_FILE=	${WRKSRC}/LICENSE

OPTIONS_DEFINE=	DOCS
OPTIONS_SUB=	yes

USE_GITHUB=	yes
GH_ACCOUNT=	rime
GH_TAGNAME=	ca7df5f5498ccd99cc312413ceae7d13717456b8

NO_ARCH=	yes
NO_BUILD=	yes

PORTDOCS=	AUTHORS README.md

do-install:
	@${MKDIR} ${STAGEDIR}${PREFIX}/share/rime-data
	${INSTALL_DATA} ${WRKSRC}/emoji_suggestion.yaml ${STAGEDIR}${PREFIX}/share/rime-data
	(cd ${WRKSRC} && ${COPYTREE_SHARE} opencc ${STAGEDIR}${PREFIX}/share/rime-data)

do-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_MAN} ${WRKSRC}/AUTHORS ${STAGEDIR}/${DOCSDIR}
	${INSTALL_MAN} ${WRKSRC}/README.md ${STAGEDIR}/${DOCSDIR}

.include <bsd.port.mk>
