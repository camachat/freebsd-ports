PORTNAME=	jericho-html
PORTVERSION=	3.2
CATEGORIES=	www java
MASTER_SITES=	SF/jerichohtml/${PORTNAME}/${PORTVERSION}

MAINTAINER=	saper@saper.info
COMMENT=	Java library to analyse and manipulate HTML
WWW=		https://jerichohtml.sourceforge.net/doc/index.html

USES=		java zip

INTERFACES:=	"compile-time-dependencies/slf4j-api-1.6.1.jar:\
		compile-time-dependencies/commons-logging-api-1.1.1.jar:\
		compile-time-dependencies/log4j-1.2.16.jar"

PORTDOCS=	*
PLIST_FILES+=	%%JAVAJARDIR%%/${PORTNAME}.jar

OPTIONS_DEFINE=	DOCS

do-build:
	(cd ${WRKSRC} && ${RM} -r classes/* && ${SETENV}         \
		LANG=en_US.ISO8859-1 LC_ALL=en_US.ISO8859-1 ${JAVAC} \
		-classpath ${INTERFACES:S, ,,g}                   \
		-d classes src/java/net/htmlparser/jericho/*.java \
		src/java/net/htmlparser/jericho/nodoc/*.java)
	${MKDIR} ${WRKSRC}/lib
	${JAR} -cf ${WRKSRC}/lib/${PORTNAME}.jar                  \
		 -C ${WRKSRC}/classes .

do-install:
	${MKDIR} ${STAGEDIR}${JAVAJARDIR}
	${INSTALL_DATA} ${WRKSRC}/lib/${PORTNAME}.jar ${STAGEDIR}${JAVAJARDIR}
	${MKDIR} ${STAGEDIR}${DOCSDIR}
	(cd ${WRKSRC}/docs && ${COPYTREE_SHARE} . ${STAGEDIR}${DOCSDIR})

.include <bsd.port.mk>
