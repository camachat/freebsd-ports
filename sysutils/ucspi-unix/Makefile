PORTNAME=	ucspi-unix
PORTVERSION=	1.0
PORTREVISION=	1
CATEGORIES=	sysutils
MASTER_SITES=	http://untroubled.org/${PORTNAME}/archive/

MAINTAINER=	ports@FreeBSD.org
COMMENT=	Unix-domain socket client-server command-line tools
WWW=		https://untroubled.org/ucspi-unix/

LICENSE=	GPLv2

BUILD_DEPENDS=	${BGLIBS_LIB}/libbg.a:devel/bglibs

USES=		libtool:build localbase:ldflags
BGLIBS_INC=	${LOCALBASE}/include/bglibs
BGLIBS_LIB=	${LOCALBASE}/lib/bglibs

PLIST_FILES=	bin/unixcat bin/unixclient bin/unixserver \
		share/man/man1/unixclient.1.gz share/man/man1/unixserver.1.gz
PORTDOCS=	NEWS PROTOCOL README TODO

OPTIONS_DEFINE=	DOCS

MAKE_JOBS_UNSAFE=yes

post-patch: .SILENT
	${ECHO_CMD} "${STAGEDIR}${PREFIX}"		> ${WRKSRC}/conf-home
	${ECHO_CMD} "${STAGEDIR}${PREFIX}/bin"	> ${WRKSRC}/conf-bin
	${ECHO_CMD} "${STAGEDIR}${PREFIX}/share/man"	> ${WRKSRC}/conf-man
	${ECHO_CMD} "${CC} ${CFLAGS}"	> ${WRKSRC}/conf-cc
	${ECHO_CMD} "${CC} ${LDFLAGS} ${STRIP}"	> ${WRKSRC}/conf-ld
	${ECHO_CMD} "${BGLIBS_INC}"	> ${WRKSRC}/conf-bgincs
	${ECHO_CMD} "${BGLIBS_LIB}"	> ${WRKSRC}/conf-bglibs

do-install:
	${INSTALL_SCRIPT} ${WRKSRC}/unixcat ${STAGEDIR}${PREFIX}/bin/
	${INSTALL_PROGRAM} ${WRKSRC}/unixclient ${STAGEDIR}${PREFIX}/bin/
	${INSTALL_PROGRAM} ${WRKSRC}/unixserver ${STAGEDIR}${PREFIX}/bin/
	${INSTALL_MAN} ${WRKSRC}/unixclient.1 ${STAGEDIR}${PREFIX}/share/man/man1/
	${INSTALL_MAN} ${WRKSRC}/unixserver.1 ${STAGEDIR}${PREFIX}/share/man/man1/

do-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${PORTDOCS:S|^|${WRKSRC}/|} ${STAGEDIR}${DOCSDIR}

.include <bsd.port.mk>
