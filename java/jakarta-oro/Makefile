PORTNAME=	jakarta-oro
PORTVERSION=	2.0.8
PORTREVISION=	3
CATEGORIES=	java
MASTER_SITES=	http://archive.apache.org/dist/jakarta/oro/

MAINTAINER=	ports@FreeBSD.org
COMMENT=	Perl/awk/sh regular expressions for Java
WWW=		https://jakarta.apache.org/oro/

LICENSE=	APACHE20

USES=		java:ant,run

ALL_TARGET=	jar javadocs

JAVALIBNAME=	${PORTNAME}-${PORTVERSION}

PLIST_FILES=	%%JAVAJARDIR%%/${PORTNAME}.jar
PORTDOCS=	*

OPTIONS_DEFINE=	DOCS

do-install:
	@${MKDIR} ${STAGEDIR}${JAVAJARDIR}
	${INSTALL_DATA} ${WRKSRC}/${PORTNAME}-${PORTVERSION}.jar \
		${STAGEDIR}${JAVAJARDIR}/${PORTNAME}.jar
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	(cd ${WRKSRC}/docs/api && ${COPYTREE_SHARE} . ${STAGEDIR}${DOCSDIR})

.include <bsd.port.mk>
