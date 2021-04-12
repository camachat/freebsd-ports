# Created by: nork@FreeBSD.org
<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	openexr
PORTVERSION=	2.5.5
=======

PORTNAME=	openexr
PORTVERSION=	3.0.1
>>>>>>> upstream/main
CATEGORIES=	graphics devel
MASTER_SITES=	LOCAL/mandree/:test
DISTFILES=	${PORTNAME}-2.2.0-comp_dwa-test-images.tar.xz:test

MAINTAINER=	mandree@FreeBSD.org
COMMENT=	High dynamic-range (HDR) image file format

LICENSE=	BSD3CLAUSE

# exact version required to avoid hard-to-debug issues
<<<<<<< HEAD
LIB_DEPENDS=	libImath-2_5.so.25:graphics/ilmbase

USES=		cmake compiler:c++14-lang libtool pathfix pkgconfig
=======
LIB_DEPENDS=	libImath-3_0.so.27:math/Imath

USES=		cmake compiler:c++14-lang pathfix pkgconfig
>>>>>>> upstream/main

USE_GITHUB=	yes
GH_TUPLE=	AcademySoftwareFoundation:openexr:v${PORTVERSION}

USE_LDCONFIG=	yes
<<<<<<< HEAD
=======
CMAKE_ARGS+=	-DCMAKE_DEBUG_POSTFIX=
>>>>>>> upstream/main
CPPFLAGS+=	-I. -I../IlmImf
# must be linked with -l{thr|pthread} explicitly
LDFLAGS+=	-lpthread

<<<<<<< HEAD
WRKSRC_SUBDIR=	OpenEXR
=======
MAJORVER=	3_0
_VER=		27
_MINVER=	0
_PLVER=		0

PLIST_SUB+=	MAJORVER=${MAJORVER} \
		VER=${_VER} \
		MINVER=${_MINVER} \
		PLVER=${_PLVER}
>>>>>>> upstream/main

PORTDOCS=	*
PORTEXAMPLES=	*

OPTIONS_DEFINE=	DOCS EXAMPLES
OPTIONS_SUB=	yes

<<<<<<< HEAD
_DOCSRCDIR1=	${WRKSRC:H}/
_DOCSRCDIR2=	${WRKSRC}/doc
=======
_DOCSRCDIR1=	${WRKSRC}
_DOCSRCDIR2=	${WRKSRC}/docs
>>>>>>> upstream/main
_DOC_FILES1=	CHANGES.md CONTRIBUTING.md GOVERNANCE.md LICENSE.md SECURITY.md \
		CODE_OF_CONDUCT.md CONTRIBUTORS.md README.md
_DOC_FILES2=	*.odt *.pdf

<<<<<<< HEAD
MAJORVER=	2_5
_VER=		25
_MINVER=	0
_PLVER=		4

PLIST_SUB+=	MAJORVER=${MAJORVER} \
		VER=${_VER} \
		MINVER=${_MINVER} \
		PLVER=${_PLVER}

# too many reports about compilation failures, so
# sanity check we are using the same C++ standard library
_ilm_libcxx=	${COMPILER_FEATURES:Mlib*c++}
pre-configure:
	@${READELF} -d ${LOCALBASE}/lib/libImath.so \
	| ${EGREP} -q '\<NEEDED\>.*\[${_ilm_libcxx:C/\+/\\+/g}\.' \
	|| {	${ECHO_CMD} "*** Your ilmbase package uses a different C++ standard library than ***" ; \
		${ECHO_CMD} "*** OpenEXR would. Please recompile and reinstall ilmbase with the  ***" ; \
		${ECHO_CMD} "*** same C++ std. library before trying to build OpenEXR.  Abort.   ***" ; \
		exit 1; }

post-install:
	${STRIP_CMD} \
		${STAGEDIR}${PREFIX}/lib/libIlmImf-${MAJORVER}.so.${_VER} \
		${STAGEDIR}${PREFIX}/lib/libIlmImfUtil-${MAJORVER}.so.${_VER}

post-install-DOCS-off:
	${RM} ${STAGEDIR}${DOCSDIR:H}/OpenEXR/*.pdf
	@${RMDIR} ${STAGEDIR}${PREFIX}/share/doc/${PORTNAME} 2>/dev/null || :

=======
# too many reports about compilation failures, so
# sanity check we are using the same C++ standard library
_imath_libcxx=	${COMPILER_FEATURES:Mlib*c++}
pre-configure:
	@${READELF} -d ${LOCALBASE}/lib/libImath.so \
	| ${EGREP} -q '\<NEEDED\>.*\[${_imath_libcxx:C/\+/\\+/g}\.' \
	|| {	${ECHO_CMD} "*** Your Imath package uses a different C++ standard library than ***" ; \
		${ECHO_CMD} "*** OpenEXR would. Please recompile and reinstall Imath with the  ***" ; \
		${ECHO_CMD} "*** same C++ std. library before trying to build OpenEXR.  Abort. ***" ; \
		exit 1; }

>>>>>>> upstream/main
post-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${_DOC_FILES1:S|^|${_DOCSRCDIR1}/|} ${STAGEDIR}${DOCSDIR}
	${INSTALL_DATA} ${_DOC_FILES2:S|^|${_DOCSRCDIR2}/|} ${STAGEDIR}${DOCSDIR}
<<<<<<< HEAD
	${MV} ${STAGEDIR}${DOCSDIR:H}/OpenEXR/[a-df-zA-Z]* ${STAGEDIR}${DOCSDIR}

post-install-EXAMPLES-off:
	${RM} -R ${STAGEDIR}${PREFIX}/share/doc/OpenEXR/examples/
	@${RMDIR} ${STAGEDIR}${PREFIX}/share/doc/OpenEXR 2>/dev/null || :
=======
	#${MV} ${STAGEDIR}${DOCSDIR:H}/OpenEXR/[a-df-zA-Z]* ${STAGEDIR}${DOCSDIR}

post-install-DOCS-off:
	${RM} ${STAGEDIR}${DOCSDIR:H}/OpenEXR/*.pdf
	@${RMDIR} ${STAGEDIR}${PREFIX}/share/doc/${PORTNAME} 2>/dev/null || :
>>>>>>> upstream/main

post-install-EXAMPLES-on:
	${MV} \
		${STAGEDIR}${PREFIX}/share/doc/OpenEXR/examples/ ${STAGEDIR}${EXAMPLESDIR}
	@${RMDIR} ${STAGEDIR}${PREFIX}/share/doc/OpenEXR 2>/dev/null || :

<<<<<<< HEAD
=======
post-install-EXAMPLES-off:
	${RM} -R ${STAGEDIR}${PREFIX}/share/doc/OpenEXR/examples/
	@${RMDIR} ${STAGEDIR}${PREFIX}/share/doc/OpenEXR 2>/dev/null || :

>>>>>>> upstream/main
do-test:
	cd ${BUILD_WRKSRC} && ctest -j ${MAKE_JOBS_NUMBER}

.include <bsd.port.mk>
