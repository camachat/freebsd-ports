PORTNAME=	convey
PORTVERSION=	0.3
PORTREVISION=	4
CATEGORIES=	net-im java
MASTER_SITES=	http://convey.sourceforge.net/downloads/ \
		http://nivi.interfree.it/tmp/convey.sourceforge.net/
DISTNAME=	${PORTNAME}-src-${PORTVERSION}

MAINTAINER=	nivit@FreeBSD.org
COMMENT=	Instant Message (IM) program using the Jabber protocol
WWW=		https://convey.sourceforge.net/

USES=		java:ant,run

MAKE_ARGS=	-quiet

DATADIR=	${JAVASHAREDIR}/${PORTNAME}
FIND_DIRS=	lib

SUB_FILES=	convey.sh

do-install:
	${INSTALL_SCRIPT} ${WRKDIR}/convey.sh ${STAGEDIR}${PREFIX}/bin/convey
	cd ${WRKSRC} && ${FIND} ${FIND_DIRS} -type d -exec ${MKDIR} ${STAGEDIR}${DATADIR}/{} \;
	cd ${WRKSRC} && ${FIND} ${FIND_DIRS} -type f -exec ${INSTALL_DATA} {} ${STAGEDIR}${DATADIR}/{} \;

.include <bsd.port.mk>
