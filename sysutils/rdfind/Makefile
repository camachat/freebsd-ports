PORTNAME=	rdfind
DISTVERSION=	1.7.0
CATEGORIES=	sysutils
MASTER_SITES=	https://rdfind.pauldreik.se/

MAINTAINER=	dtxdf@FreeBSD.org
COMMENT=	Utility to find duplicate files based on their content
WWW=		https://rdfind.pauldreik.se/

LICENSE=	GPLv2
LICENSE_FILE=	${WRKSRC}/COPYING

LIB_DEPENDS=	libnettle.so:security/nettle

USES=		autoreconf localbase:ldflags
GNU_CONFIGURE=	yes
GNU_CONFIGURE_MANPREFIX=${PREFIX}/share

PLIST_FILES=	bin/${PORTNAME} \
		share/man/man1/${PORTNAME}.1.gz

post-install:
	${STRIP_CMD} ${STAGEDIR}${PREFIX}/bin/${PORTNAME}

.include <bsd.port.mk>
