PORTNAME=	libada
DISTVERSIONPREFIX=	v
DISTVERSION=	3.2.6
CATEGORIES=	devel www

MAINTAINER=	osa@FreeBSD.org
COMMENT=	WHATWG-compliant and fast URL parser
WWW=		https://ada-url.com/

LICENSE=	APACHE20 MIT
LICENSE_COMB=	dual
LICENSE_FILE_APACHE20=	${WRKSRC}/LICENSE-APACHE
LICENSE_FILE_MIT=	${WRKSRC}/LICENSE-MIT

USES=		cmake localbase:ldflags
USE_LDCONFIG=	yes

CMAKE_OFF=	ADA_TESTING ADA_BENCHMARKS ADA_TOOLS
CMAKE_ON=	BUILD_SHARED_LIBS \
		CMAKE_DISABLE_FIND_PACKAGE_Git \
		CMAKE_DISABLE_FIND_PACKAGE_Python3

USE_GITHUB=	yes
GH_ACCOUNT=	ada-url
GH_PROJECT=	ada

PLIST_SUB=	SOVERSION=${PORTVERSION}

.include <bsd.port.mk>
