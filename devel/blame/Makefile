PORTNAME=	blame
DISTVERSION=	1.4-20250101
CATEGORIES=	devel
MASTER_SITES=	https://invisible-island.net/archives/rcs-blame/

MAINTAINER=	fuz@FreeBSD.org
COMMENT=	Display last modification for each line in an RCS file
WWW=		https://invisible-island.net/rcs-blame/

LICENSE=	GPLv2
LICENSE_FILE=	${WRKSRC}/COPYING

USES=		tar:tgz
GNU_CONFIGURE=	yes
TEST_TARGET=	check
PLIST_FILES=	bin/blame \
		share/man/man1/blame.1.gz

.include <bsd.port.mk>
