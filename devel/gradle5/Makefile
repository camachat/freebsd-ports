PORTNAME=	gradle
PORTVERSION=	5.0
CATEGORIES=	devel java
MASTER_SITES=	https://downloads.gradle.org/distributions/
EXTRACT_SUFX=	-bin.zip
PKGNAMESUFFIX=	5

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Project automation tool
WWW=		https://www.gradle.org/

LICENSE=	APACHE20

USES=		cpe java zip
NO_ARCH=	yes
NO_BUILD=	yes
DATADIR=	${PREFIX}/share/${PORTNAME}${PKGNAMESUFFIX}
PORTSCOUT=	ignore:1
PLIST_SUB=	PKGNAMESUFFIX=${PKGNAMESUFFIX}
SUB_LIST=	PORTVERSION=${PORTVERSION} \
		JAVA=${JAVA}
SUB_FILES=	gradle

do-install:
	${INSTALL_SCRIPT} ${WRKDIR}/gradle ${STAGEDIR}${PREFIX}/bin/gradle${PKGNAMESUFFIX}
	(cd ${WRKSRC} && ${COPYTREE_SHARE} lib ${STAGEDIR}${DATADIR})

.include <bsd.port.mk>
