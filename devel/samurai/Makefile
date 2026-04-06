PORTNAME=	samurai
DISTVERSION=	1.3
CATEGORIES=	devel

MAINTAINER=	adamw@FreeBSD.org
COMMENT=	Ninja-compatible build tool written in C
WWW=		https://github.com/michaelforney/samurai

LICENSE=	APACHE20 MIT
LICENSE_COMB=	multi
LICENSE_FILE_APACHE20=	${WRKSRC}/LICENSE
LICENSE_FILE_MIT=	${WRKSRC}/LICENSE

USES=		cpe
CPE_VENDOR=	${PORTNAME}_project

USE_GITHUB=	yes
GH_ACCOUNT=	michaelforney
#GH_TAGNAME=	4288e71

ALL_TARGET=	samu
PLIST_FILES=	bin/samu \
		share/man/man1/samu.1.gz

.include <bsd.port.mk>
