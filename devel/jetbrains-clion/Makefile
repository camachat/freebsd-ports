<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	clion
PORTVERSION=	2020.3.3
=======
PORTNAME=	clion
PORTVERSION=	2021.1
>>>>>>> upstream/main
CATEGORIES=	devel java
MASTER_SITES=	https://download-cf.jetbrains.com/cpp/
PKGNAMEPREFIX=	jetbrains-
DISTNAME=	CLion-${PORTVERSION}
DIST_SUBDIR=	jetbrains

MAINTAINER=	dmitry.wagin@ya.ru
COMMENT=	JetBrains CLion IDE

LICENSE=	IntelliJ-CLion
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

<<<<<<< HEAD
SHEBANG_FILES=	bin/printenv.py bin/restart.py
=======
SHEBANG_FILES=	bin/printenv.py bin/restart.py \
		plugins/python-ce/helpers/typeshed/scripts/generate_proto_stubs.sh
>>>>>>> upstream/main

NO_ARCH=	yes
NO_BUILD=	yes
SUB_FILES=	${PORTNAME} ${PORTNAME}.desktop pkg-message

WRKSRC=		${WRKDIR}/clion-${PORTVERSION}

do-install:
<<<<<<< HEAD
# Initial JDK 15+ support
	@${REINPLACE_CMD} '/UseConcMarkSweepGC/d' ${WRKSRC}/bin/${PORTNAME}.vmoptions
	@${REINPLACE_CMD} '/UseConcMarkSweepGC/d' ${WRKSRC}/bin/${PORTNAME}64.vmoptions
=======
>>>>>>> upstream/main
# Linux/Windows/OS X only so remove them
	@${RM} -r \
		${WRKSRC}/bin/clang \
		${WRKSRC}/bin/cmake \
		${WRKSRC}/bin/fsnotifier \
		${WRKSRC}/bin/fsnotifier-arm \
		${WRKSRC}/bin/fsnotifier64 \
		${WRKSRC}/bin/gdb/linux \
		${WRKSRC}/bin/lldb/linux \
		${WRKSRC}/jbr \
		${WRKSRC}/jre64 \
		${WRKSRC}/lib/pty4j-native \
<<<<<<< HEAD
=======
		${WRKSRC}/plugins/cwm-plugin/quiche-native \
>>>>>>> upstream/main
		${WRKSRC}/plugins/performanceTesting/bin
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
