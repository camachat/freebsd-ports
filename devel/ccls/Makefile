<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	ccls
DISTVERSION=	0.20201219
CATEGORIES=	devel

MAINTAINER=	khng300@gmail.com
=======
PORTNAME=	ccls
DISTVERSION=	0.20210330
CATEGORIES=	devel

MAINTAINER=	khng@FreeBSD.org
>>>>>>> upstream/main
COMMENT=	C/C++/ObjC language server

LICENSE=	APACHE20
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	rapidjson>=1.1.0_4:devel/rapidjson
LIB_DEPENDS=	libclang-cpp.so:devel/llvm${LLVM_DEFAULT}

USES=		cmake compiler:c++17-lang localbase
USE_GITHUB=	yes
GH_ACCOUNT=	MaskRay

CMAKE_ARGS=	-DCCLS_VERSION:STRING="${DISTVERSION}" \
		-DClang_DIR:PATH="${LOCALBASE}/llvm${LLVM_DEFAULT}/lib/cmake/clang" \
		-DLLVM_DIR:PATH="${LOCALBASE}/llvm${LLVM_DEFAULT}/lib/cmake/llvm"
CXXFLAGS+=	-DNDEBUG

PLIST_FILES=	bin/ccls
PORTDOCS=	*

OPTIONS_DEFINE=	DOCS

<<<<<<< HEAD
DOCS_MASTER_SITES=	LOCAL/tobik:docs
DOCS_DISTFILES=		MaskRay-ccls.wiki-20190903.tar.gz:docs
=======
DOCS_MASTER_SITES=	LOCAL/khng:docs
DOCS_DISTFILES=		ccls.wiki-${DISTVERSION}.tar.gz:docs
>>>>>>> upstream/main

post-install-DOCS-on:
	@${MKDIR} ${STAGEDIR}${DOCSDIR}
	cd ${WRKDIR}/ccls.wiki && ${COPYTREE_SHARE} . ${STAGEDIR}${DOCSDIR}

<<<<<<< HEAD
=======
.if defined(MAINTAINER_MODE)
MAINTAINER_DISTFILE=	\
	${_DISTDIR:S|/$||}/${DOCS_DISTFILES:Mccls.wiki-*.tar.gz\:*:C|\:.*$||}

maintainer-docs-fetch:
	@${MAKE} clean
	@${MKDIR} ${WRKDIR}/ccls.wiki
	@git clone https://github.com/MaskRay/ccls.wiki.git \
		${WRKDIR}/ccls.wiki
	@${TAR} -zcf ${MAINTAINER_DISTFILE} -C ${WRKDIR} \
		--uid 0 --gid 0 --exclude-vcs ccls.wiki
	@${MAKE} makesum
	@${MAKE} clean

${MAINTAINER_DISTFILE}: maintainer-docs-fetch

maintainer-docs-distfiles: ${MAINTAINER_DISTFILE}
	scp ${MAINTAINER_DISTFILE} \
	    freefall.freebsd.org:public_distfiles/
.endif

>>>>>>> upstream/main
.include <bsd.port.mk>
