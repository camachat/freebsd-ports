PORTNAME=	spirv-headers
DISTVERSIONPREFIX=	vulkan-sdk-
DISTVERSION=	1.4.341.0
CATEGORIES=	graphics

MAINTAINER=	tagattie@FreeBSD.org
COMMENT=	SPIRV-Headers
WWW=		https://github.com/KhronosGroup/SPIRV-Headers

LICENSE=	APACHE20
LICENSE_FILE=	${WRKSRC}/LICENSE

#USES=		cmake:testing compiler:c++11-lib python:build shebangfix
USES=		cmake:testing

USE_GITHUB=	yes
GH_ACCOUNT=	KhronosGroup
GH_PROJECT=	SPIRV-Headers

CMAKE_ON=		SPIRV_HEADERS_ENABLE_INSTALL
CMAKE_TESTING_ON=	SPIRV_HEADERS_ENABLE_TESTS
CMAKE_TESTING_OFF=	SPIRV_HEADERS_ENABLE_TESTS

.include <bsd.port.mk>
