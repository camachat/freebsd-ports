PORTNAME=	libopentracing
PORTVERSION=	1.6.0
DISTVERSIONPREFIX=	v
CATEGORIES=	devel

MAINTAINER=	ports@FreeBSD.org
COMMENT=	C++ implementation of the OpenTracing API
WWW=		https://github.com/opentracing/opentracing-cpp

LICENSE=	APACHE20
LICENSE_FILE=	${WRKSRC}/LICENSE

DEPRECATED=	https://github.com/opentracing/specification/issues/163
EXPIRATION_DATE=2024-02-29

USE_GITHUB=	yes
GH_ACCOUNT=	opentracing
GH_PROJECT=	opentracing-cpp

USES=		cmake compiler:c++11-lang

USE_LDCONFIG=	yes
GNU_CONFIGURE=	yes

PLIST_SUB=	SHLIBVER=${PORTVERSION}

post-install:
	${LN} -sf libopentracing_mocktracer.so.${PORTVERSION} \
		${STAGEDIR}${PREFIX}/lib/libopentracing_mocktracer.so.1
	${LN} -sf libopentracing_mocktracer.so.1 \
		${STAGEDIR}${PREFIX}/lib/libopentracing_mocktracer.so
	${LN} -sf libopentracing.so.${PORTVERSION} \
		${STAGEDIR}${PREFIX}/lib/libopentracing.so.1
	${LN} -sf libopentracing.so.1 \
		${STAGEDIR}${PREFIX}/lib/libopentracing.so

.include <bsd.port.mk>
