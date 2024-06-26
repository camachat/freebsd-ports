PORTNAME=	newsx
PORTVERSION=	1.6
PORTREVISION=	5
CATEGORIES=	news
MASTER_SITES=	LOCAL/thierry

MAINTAINER=	thierry@FreeBSD.org
COMMENT=	Program to fetch news from and post news to a remote NNTP server
WWW=		https://web.archive.org/web/20220208213916/http://www.kvaleberg.no/newsx.html

LICENSE=	GPLv2

.if defined(WITH_CNEWS)
BUILD_DEPENDS=	${LOCALBASE}/news/lib/libcnews.a:news/cnews
.elif !defined(WITHOUT_INN_2-X)
BUILD_DEPENDS=	${LOCALBASE}/news/lib/libinn.so:news/inn
RUN_DEPENDS=	${LOCALBASE}/news/lib/libinn.so:news/inn
.endif

USES=		cpe
GNU_CONFIGURE=	yes
GNU_CONFIGURE_MANPREFIX=${PREFIX}/share
CFLAGS+=	-fcommon

CONFLICTS=	leafnode-1.*

OPTIONS_DEFINE=	DOCS

.if !defined(WITH_CNEWS)
.if !defined(WITHOUT_INN_2-X)
CPPFLAGS+=	-I${LOCALBASE}/news/include/inn
.endif
.endif

.if defined(PACKAGE_BUILDING)
CONFIGURE_ARGS+=	--without-history
.endif

PORTDOCS=		AUTHORS COPYING ChangeLog FAQ INSTALL NEWS README TODO

.include <bsd.port.options.mk>

pre-configure:
	${REINPLACE_CMD} -e "s:%%LOCALBASE%%:${LOCALBASE}:g" ${WRKSRC}/configure
	@${ECHO} "*********************************************************"
	@${ECHO} "If the configure step fails, then you probably don't have"
	@${ECHO} "a news server package installed.  Try installing INN from"
	@${ECHO} "${PORTSDIR}/news/inn, and then try building this port"
	@${ECHO} "again."
	@${ECHO} "*********************************************************"

post-build:
.if !defined(WITH_DEBUG)
	${STRIP_CMD} ${WRKSRC}/src/newsx
.endif

post-install-DOCS-on:
	${MKDIR} ${STAGEDIR}${DOCSDIR}
.for FILE in ${PORTDOCS}
	${INSTALL_DATA} ${WRKSRC}/${FILE} ${STAGEDIR}${DOCSDIR}
.endfor

.include <bsd.port.mk>
