PORTNAME=	plantuml
PORTVERSION=	1.2025.4
CATEGORIES=	graphics java textproc
MASTER_SITES=	https://github.com/plantuml/plantuml/releases/download/v${PORTVERSION}/
EXTRACT_SUFX=	.jar
EXTRACT_ONLY=

MAINTAINER=	awoonyaa@gmail.com
COMMENT=	Application for swift and straightforward diagram creation
WWW=		https://plantuml.com/

LICENSE=	GPLv3

USES=		java

NO_ARCH=	yes
NO_BUILD=	yes

PLIST_FILES=	%%JAVASHAREDIR%%/${PORTNAME}/${PORTNAME}.jar bin/${PORTNAME}

pre-install:
	${ECHO} "#!/bin/sh" > ${WRKDIR}/plantum.sh
	${ECHO} '${PREFIX}/bin/java -jar ${JAVASHAREDIR}/${PORTNAME}/${PORTNAME}.jar $$*' >> ${WRKDIR}/plantum.sh

do-install:
	${MKDIR} ${STAGEDIR}${JAVASHAREDIR}/${PORTNAME}
	${INSTALL_DATA} ${DISTDIR}/${PORTNAME}-${PORTVERSION}.jar ${STAGEDIR}${JAVASHAREDIR}/${PORTNAME}/${PORTNAME}.jar
	${INSTALL_SCRIPT} ${WRKDIR}/plantum.sh ${STAGEDIR}${PREFIX}/bin/${PORTNAME}

.include <bsd.port.mk>
