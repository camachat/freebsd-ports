PORTNAME=	multitail
PORTVERSION=	6.5.0
PORTREVISION=	1
CATEGORIES=	sysutils
MASTER_SITES=	http://www.vanheusden.com/multitail/ \
		http://fossies.org/unix/privat/

MAINTAINER=	beat@FreeBSD.org
COMMENT=	Tail multiple files on console with ncurses
WWW=		https://www.vanheusden.com/multitail/

LICENSE=	GPLv2
LICENSE_FILE=	${WRKSRC}/license.txt

USES=		alias gmake ncurses tar:tgz

PLIST_FILES=	bin/multitail etc/multitail.conf.sample \
		share/man/man1/multitail.1.gz
PORTDOCS=	readme.txt license.txt
MAKE_ARGS=	NCURSES_IMPL="${NCURSES_IMPL}"

OPTIONS_DEFINE=	DOCS

do-install:
	${INSTALL_PROGRAM} ${WRKSRC}/multitail ${STAGEDIR}${PREFIX}/bin/
	${INSTALL_MAN} ${WRKSRC}/multitail.1 ${STAGEDIR}${PREFIX}/share/man/man1
	${INSTALL_DATA} ${WRKSRC}/multitail.conf ${STAGEDIR}${PREFIX}/etc/multitail.conf.sample
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${WRKSRC}/readme.txt ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${WRKSRC}/license.txt ${STAGEDIR}${DOCSDIR}

.include <bsd.port.mk>
