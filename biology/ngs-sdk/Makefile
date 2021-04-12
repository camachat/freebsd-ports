<<<<<<< HEAD
# $FreeBSD$

PORTNAME=	ngs
DISTVERSION=	2.11.0
=======
PORTNAME=	ngs
DISTVERSION=	2.11.0
PORTREVISION=	1
>>>>>>> upstream/main
CATEGORIES=	biology
PKGNAMESUFFIX=	-sdk

MAINTAINER=	yuri@FreeBSD.org
COMMENT=	Tools and libraries for using data in the INSDC Sequence Read Archives

LICENSE=	PD
LICENSE_FILE=	${WRKSRC}/LICENSE

BROKEN_aarch64=		fails to configure: unsupported architecture 'FreeBSD'
BROKEN_riscv64=		fails to configure: unsupported architecture 'FreeBSD'
NOT_FOR_ARCHS=		powerpc64
NOT_FOR_ARCHS_REASON=	os-arch.prl needs to be taught about powerpc64

<<<<<<< HEAD
USES=		gmake perl5
GNU_CONFIGURE=	yes
USE_GITHUB=	yes
GH_ACCOUNT=	ncbi
ALL_TARGET=	default
USE_LDCONFIG=	yes

CFLAGS+=	-I${WRKSRC}/ngs-sdk/ngs/unix/${ARCH}/

post-patch:
	@cd ${WRKSRC}/ngs-sdk/ngs/unix && ${LN} -s x86_64 amd64

do-install:
	cd ${WRKSRC}/ngs-sdk && ${COPYTREE_SHARE} ngs ${STAGEDIR}/${PREFIX}/include
	${INSTALL_PROGRAM} ${WRKSRC}/ngs-sdk/${ARCH}-portbld-*/ngs-sdk/FreeBSD/clang/${ARCH}/rel/lib/libngs-sdk.so.${DISTVERSION} ${STAGEDIR}${PREFIX}/lib/
	cd ${STAGEDIR}${PREFIX}/lib/ && \
		${LN} -s libngs-sdk.so.${DISTVERSION} libngs-sdk.so.${DISTVERSION:R:R} && \
		${LN} -s libngs-sdk.so.${DISTVERSION:R:R} libngs-sdk.so
.for l in libngs-c++.a libngs-adapt-c++.a
	${INSTALL_LIB} ${WRKSRC}/ngs-sdk/${ARCH}-portbld-*/ngs-sdk/FreeBSD/clang/${ARCH}/rel/lib/${l}.${PORTVERSION} ${STAGEDIR}${PREFIX}/lib/${l}
.endfor
=======
USES=		compiler:c++11-lang gmake perl5
USE_PERL5=	build
USE_LDCONFIG=	yes

USE_GITHUB=	yes
GH_ACCOUNT=	ncbi

HAS_CONFIGURE=	yes
CONFIGURE_ARGS=	--prefix="${STAGEDIR}${PREFIX}" --build-prefix="${WRKSRC}/build"

ALL_TARGET=	ngs-sdk

post-extract:
	@cd ${WRKSRC}/ngs-sdk/ngs/unix && ${LN} -s x86_64 amd64

do-install: # install target from the project breaks: https://github.com/ncbi/ngs/issues/34
	# install headers
	cd ${WRKSRC}/ngs-sdk && ${COPYTREE_SHARE} ngs ${STAGEDIR}${PREFIX}/include
	${RM} -r ${STAGEDIR}${PREFIX}/include/ngs/win
	# install libraries
	cd ${WRKSRC}/build/ngs-sdk/FreeBSD/clang/amd64/rel/lib && ${COPYTREE_SHARE} . ${STAGEDIR}${PREFIX}/lib
	# remove static libs built instead of shared libs, see https://github.com/ncbi/ngs/issues/33
	cd ${STAGEDIR}${PREFIX}/lib && ${RM} libngs-c++.a* libngs-adapt-c++.a*
	# strip
	${STRIP_CMD} ${STAGEDIR}${PREFIX}/lib/libngs-sdk.so.2.11.0
>>>>>>> upstream/main

.include <bsd.port.mk>
