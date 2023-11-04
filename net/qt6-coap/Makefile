PORTNAME=	coap
DISTVERSION=	${QT6_VERSION}
CATEGORIES=	net
PKGNAMEPREFIX=	qt6-

MAINTAINER=	kde@FreeBSD.org
COMMENT=	Qt6 CoAP client module
WWW=		https://qt-project.org

USE_GITHUB=	yes
GH_TUPLE=       qt:qtcoap:${QT6_VERSION}

USES=		cmake compiler:c++17-lang qt:6
USE_QT=		base
USE_LDCONFIG=	${PREFIX}/${QT_LIBDIR_REL}

PLIST_SUB+=	SHORTVER=${_QT_VERSION:R} \
		FULLVER=${_QT_VERSION:C/-.*//}

.include <bsd.port.mk>
