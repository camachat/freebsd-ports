PORTNAME=	wasi-libc
DISTVERSION=	g20220906
CATEGORIES=	devel

MAINTAINER=	greg@unrelenting.technology
COMMENT=	C standard library for WebAssembly System Interface
WWW=		https://github.com/CraneStation/wasi-libc

LICENSE=	APACHE20 MIT LLVM2
LICENSE_COMB=	multi
LICENSE_FILE_APACHE20=	${WRKSRC}/LICENSE-APACHE
LICENSE_FILE_MIT=	${WRKSRC}/LICENSE-MIT
LICENSE_FILE_LLVM2=	${WRKSRC}/LICENSE-APACHE-LLVM
LICENSE_NAME_LLVM2=	Apache License 2.0 with LLVM Exceptions
LICENSE_PERMS_LLVM2=	dist-mirror dist-sell pkg-mirror pkg-sell auto-accept

BUILD_DEPENDS=	clang${LLVM_VERSION}:devel/llvm${LLVM_VERSION}

USES=		gmake
USE_GITHUB=	yes
GH_ACCOUNT=	WebAssembly
GH_TAGNAME=	c40403f9b02f1b71c73a0537518ba26f91c194c4
GH_TUPLE=	WebAssembly:WASI:59cbe140561db52fc505555e859de884e0ee7f00:wasi/tools/wasi-headers/WASI
NO_ARCH=	yes
LLVM_VERSION?=	${LLVM_DEFAULT}

ALL_TARGET=	finish

# NOTE: matches the https://github.com/CraneStation/wasi-sdk
WASI_SYSROOT=	${PREFIX}/share/wasi-sysroot

# NOTE: our llvm ports don't ship stdarg/stddef clang headers, so they're in FILESDIR
#       and we install them to the sysroot

MAKE_ARGS+=	INSTALL_DIR=${STAGEDIR}${WASI_SYSROOT} \
		CPP=${LOCALBASE}/bin/clang-cpp${LLVM_VERSION} \
		CC=${LOCALBASE}/bin/clang${LLVM_VERSION} \
		AR=${LOCALBASE}/bin/llvm-ar${LLVM_VERSION} \
		NM=${LOCALBASE}/bin/llvm-nm${LLVM_VERSION} \
		EXTRA_CFLAGS="-O2 -DNDEBUG -isystem ${FILESDIR}"

# NOTE: phony targets cause make install to rebuild everything
#       https://github.com/CraneStation/wasi-libc/issues/156
post-patch:
	${REINPLACE_CMD} -e 's|install: finish|install:|' ${WRKSRC}/Makefile

post-install:
	${INSTALL_DATA} ${FILESDIR}/stdarg.h ${STAGEDIR}${WASI_SYSROOT}/include/
	${INSTALL_DATA} ${FILESDIR}/stddef.h ${STAGEDIR}${WASI_SYSROOT}/include/

.include <bsd.port.mk>
