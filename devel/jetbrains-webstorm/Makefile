# Created by: Pedro Giffuni <pfg@FreeBSD.org>
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	webstorm
PORTVERSION=	2020.3.3
=======

PORTNAME=	webstorm
PORTVERSION=	2021.1
>>>>>>> upstream/main
CATEGORIES=	devel java
MASTER_SITES=	https://download-cf.jetbrains.com/webstorm/
PKGNAMEPREFIX=	jetbrains-
DISTNAME=	WebStorm-${PORTVERSION}
DIST_SUBDIR=	jetbrains

MAINTAINER=	dmitry.wagin@ya.ru
COMMENT=	JetBrains JavaScript IDE

LICENSE=	IntelliJ-WebStorm
LICENSE_NAME=	Subscription license agreement for business and organizations
LICENSE_TEXT=	See: https://www.jetbrains.com/store/license.html
LICENSE_PERMS=	dist-mirror pkg-mirror auto-accept

RUN_DEPENDS=	intellij-fsnotifier>0:java/intellij-fsnotifier \
<<<<<<< HEAD
		intellij-pty4j>=0.11.4:java/intellij-pty4j \
=======
>>>>>>> upstream/main
		lsof:sysutils/lsof

USES=		python:run shebangfix

USE_JAVA=	yes
JAVA_VERSION=	11+

SHEBANG_FILES=	bin/printenv.py bin/restart.py \
		plugins/spy-js/server/node_modules/node-uuid/benchmark/bench.sh \
		plugins/spy-js/server/proxy-nix.sh \
		plugins/spy-js/server/proxy-osx.sh

NO_ARCH=	yes
NO_BUILD=	yes
SUB_FILES=	${PORTNAME} ${PORTNAME}.desktop pkg-message

<<<<<<< HEAD
WRKSRC=		${WRKDIR}/WebStorm-203.7717.59

do-install:
# Initial JDK 15+ support
	@${REINPLACE_CMD} '/UseConcMarkSweepGC/d' ${WRKSRC}/bin/${PORTNAME}.vmoptions
	@${REINPLACE_CMD} '/UseConcMarkSweepGC/d' ${WRKSRC}/bin/${PORTNAME}64.vmoptions
=======
WRKSRC=		${WRKDIR}/WebStorm-211.6693.108

do-install:
>>>>>>> upstream/main
# Linux/Windows/OS X only so remove them
	@${RM} -r \
		${WRKSRC}/bin/fsnotifier \
		${WRKSRC}/bin/fsnotifier-arm \
		${WRKSRC}/bin/fsnotifier64 \
		${WRKSRC}/jbr \
		${WRKSRC}/jre64 \
		${WRKSRC}/lib/pty4j-native \
<<<<<<< HEAD
=======
		${WRKSRC}/plugins/cwm-plugin/quiche-native \
>>>>>>> upstream/main
		${WRKSRC}/plugins/performanceTesting/bin \
		${WRKSRC}/plugins/webp/lib/libwebp
	${MKDIR} ${STAGEDIR}${DATADIR}
	@(cd ${WRKSRC} && ${COPYTREE_SHARE} . ${STAGEDIR}${DATADIR} \
		"! -name *\.bak ! -name *\.so ! -name *\.dll ! -name *\.dylib ! -name *\.pdb ! -name *\.sh")
	@(cd ${WRKSRC} && ${COPYTREE_BIN} . ${STAGEDIR}${DATADIR} "-name *\.sh")
	${INSTALL_SCRIPT} ${WRKDIR}/${PORTNAME} ${STAGEDIR}${PREFIX}/bin/${PORTNAME}
	${INSTALL_MAN} ${FILESDIR}/${PORTNAME}.1 ${STAGEDIR}${PREFIX}/man/man1
	${INSTALL_DATA} ${WRKDIR}/${PORTNAME}.desktop ${STAGEDIR}${PREFIX}/share/applications/
# Use fsnotifier replacement provided by java/intellij-fsnotifier
	${ECHO} "idea.filewatcher.executable.path=${PREFIX}/intellij/bin/fsnotifier" >> ${STAGEDIR}${DATADIR}/bin/idea.properties
<<<<<<< HEAD
# Use pty4j replacement provided by java/intellij-pty4j
	${LN} -sf ../../../intellij/lib/libpty ${STAGEDIR}${DATADIR}/lib/pty4j-native
=======
>>>>>>> upstream/main

.include <bsd.port.mk>
