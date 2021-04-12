# Created by: Johannes Dieterich <dieterich@ogolem.org>
<<<<<<< HEAD
# $FreeBSD$
=======
>>>>>>> upstream/main

PORTNAME=	clfft
PORTVERSION=	2.12.2
DISTVERSIONPREFIX=	v
PORTREVISION=	1
CATEGORIES=	math

MAINTAINER=	jmd@FreeBSD.org
COMMENT=	Software library containing FFT functions written in OpenCL

LICENSE=	APACHE20
LICENSE_FILE=	${WRKSRC}/LICENSE

BUILD_DEPENDS=	opencl>=0:devel/opencl
LIB_DEPENDS=	libOpenCL.so:devel/ocl-icd
RUN_DEPENDS=	opencl>=0:devel/opencl

<<<<<<< HEAD
ONLY_FOR_ARCHS=	amd64
=======
ONLY_FOR_ARCHS=	amd64 powerpc64 powerpc64le
>>>>>>> upstream/main

USES=		cmake
USE_LDCONFIG=	yes
USE_CXXSTD=	gnu++98

USE_GITHUB=	yes
GH_ACCOUNT=	clMathLibraries
GH_PROJECT=	clFFT

CMAKE_ARGS+=	-DBUILD_CLIENT=OFF \
		-DBUILD_TEST=OFF \
		-DBUILD_CALLBACK_CLIENT=OFF \
		-DSUFFIX_LIB=""

CMAKE_SOURCE_PATH=	${WRKSRC}/src

.include <bsd.port.mk>
