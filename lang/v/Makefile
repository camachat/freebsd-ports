PORTNAME=	v
PORTVERSION=	2022.19
DISTVERSIONPREFIX=	weekly.
CATEGORIES=	lang

MAINTAINER=	sunpoet@FreeBSD.org
COMMENT=	V Programming Language
WWW=		https://vlang.io/

LICENSE=	MIT
LICENSE_FILE=	${WRKSRC}/LICENSE

USES=		compiler:c11 gmake

LDFLAGS+=	-lm -lexecinfo -lpthread

PLIST_FILES=	bin/v

GH_ACCOUNT=	vlang
GH_TUPLE=	vlang:vc:a298ad7069f6:vc/vc
USE_GITHUB=	yes

do-build:
	${CC} ${CFLAGS} ${LDFLAGS} ${WRKSRC}/vc/v.c -o ${WRKSRC}/v

do-install:
	${INSTALL_PROGRAM} ${WRKSRC}/v ${STAGEDIR}${PREFIX}/bin

.include <bsd.port.mk>
