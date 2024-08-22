PORTNAME=	libclc
DISTVERSION=	18.1.8
CATEGORIES=	devel
MASTER_SITES=	https://github.com/llvm/llvm-project/releases/download/llvmorg-${DISTVERSION:S/rc/-rc/}/
DISTNAME=	${PORTNAME}-${DISTVERSION}.src
DISTFILES=	${PORTNAME}-${DISTVERSION}.src${EXTRACT_SUFX}

MAINTAINER=	x11@FreeBSD.org
COMMENT=	Required library functions for OpenCL C programming language
WWW=		https://libclc.llvm.org/

LICENSE=	LLVM
LICENSE_FILE=	${WRKSRC}/LICENSE.TXT
LICENSE_NAME=	Apache License 2.0 with LLVM Exceptions
LICENSE_PERMS=	dist-mirror dist-sell pkg-mirror pkg-sell auto-accept

BUILD_DEPENDS=	spirv-llvm-translator-llvm${LLVM_SUFFIX}>0:devel/spirv-llvm-translator@llvm${LLVM_SUFFIX} \
		libedit>=0:devel/libedit

USES=		cmake compiler:c++11-lib llvm:${LLVM_SUFFIX} python:build tar:xz
USE_LDCONFIG=	yes

LLVM_RELEASE=	${DISTVERSION:C/rc.*//}
LLVM_SUFFIX=	${LLVM_RELEASE:C/\.[0-9]\.[0-9]$//}

BINARY_ALIAS=	python=${PYTHON_CMD}
CMAKE_ARGS+=	-DLLVM_CONFIG=${LOCALBASE}/bin/llvm-config${LLVM_SUFFIX}

.include <bsd.port.mk>
